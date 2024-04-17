import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
  RagRetriever, 
  RagConfig, 
  RagSequenceForGeneration,
  RagTokenForGeneration,
  AutoModelForSeq2SeqLM, 
  AutoTokenizer, 
  AutoModel, 
  MBartConfig, 
  XLMRobertaConfig, 
  Seq2SeqTrainer,
  Seq2SeqTrainingArguments
)

parser = argparse.ArgumentParser(description="train RAG")
parser.add_argument("--docs_dir", type=str, default="./data/RAG_document", help="docs directory")
parser.add_argument("--faiss_path", type=str, default="./data/RAG_document.faiss", help="docs faiss path")
parser.add_argument("--n_docs", type=int, default=3, help="number of docs to retrieve")
parser.add_argument("--train_path", type=str, default="./data/law_train_data.jsonl", help="train data path")
parser.add_argument("--valid_path", type=str, default="./data/law_valid_data.jsonl", help="valid data path")
parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--max_steps", type=int, default=1000, help="max steps")
parser.add_argument("--save_steps", type=int, default=500, help="save steps")
parser.add_argument("--eval_steps", type=int, default=1000, help="eval steps")
parser.add_argument("--logging_steps", type=int, default=5, help="logging steps")
parser.add_argument("--model_saving_path", type=str, default="./trained_models/rag", help="model saving path")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device : {device}")

encoder_id = "sentence-transformers/xlm-r-base-en-ko-nli-ststb"
encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_id)
encoder = AutoModel.from_pretrained(encoder_id).to(device)

generator_id = "facebook/mbart-large-50"
generator_tokenizer = AutoTokenizer.from_pretrained(generator_id, src_lang="ko_KR", tgt_lang="ko_KR")
generator = AutoModelForSeq2SeqLM.from_pretrained(generator_id).to(device)

generator_config = MBartConfig.get_config_dict(generator_id)[0]
encoder_config = XLMRobertaConfig.get_config_dict(encoder_id)[0]
encoder_config['model_type'] = "xlm-roberta"
rag_config = RagConfig(
  index_name="custom", 
  generator=generator_config, 
  question_encoder=encoder_config, 
  retrieval_vector_size=1024,
  passages_path=args.docs_dir,
  index_path=args.faiss_path,
  n_docs=args.n_docs,
)

retriever = RagRetriever(rag_config, encoder_tokenizer, generator_tokenizer)
model = RagTokenForGeneration(rag_config, encoder, generator, retriever)
train_ds = Dataset.from_json(args.train_path)
valid_ds = Dataset.from_json(args.valid_path)

def preprocess_data(examples):
    input_sequence = f"""
        다음 내용을 보고 참고해야 할 법률을 찾아주세요.
        {examples['context']}
    """
    input_sequence = examples['context']
    targets = ""
    for idx, t in enumerate(examples['target']):
        targets += f"{idx+1}. {t} "

    prompt = f"""
        참고해야 할 법은 {targets}등이 있습니다.
    """

    encoder_input = encoder_tokenizer(input_sequence, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    generator_input = generator_tokenizer(prompt, return_tensors="pt", max_length=512, padding="max_length", truncation=True)

    return {
        'input_ids': encoder_input.input_ids.squeeze(),
        'labels': generator_input.input_ids.squeeze(),
        'attention_mask': encoder_input.attention_mask.squeeze(),
        'decoder_attention_mask': generator_input.attention_mask.squeeze()
    }

encoded_tds = train_ds.map(preprocess_data, remove_columns=train_ds.column_names)
encoded_vds = valid_ds.map(preprocess_data, remove_columns=valid_ds.column_names)

class RAG(nn.Module):
    def __init__(self, model):
        super(RAG, self).__init__()
        self.model = model
        self.criterian = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels, attention_mask, decoder_attention_mask, **kwargs):
        # Encode
        question_hidden_states = self.model.question_encoder(input_ids, attention_mask)[0].mean(1).cpu()

        # Retrieve
        docs_dict = retriever(input_ids.cpu().numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
        doc_scores = torch.bmm(
            question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
        ).squeeze(1)

        # Forward to generator
        outputs = self.model(
            context_input_ids=docs_dict["context_input_ids"].to(device),
            context_attention_mask=docs_dict["context_attention_mask"].to(device),
            doc_scores=doc_scores,
            decoder_input_ids=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        shifted_labels = torch.cat([labels[:, 1:], torch.full((labels.size(0), 1), 1, device=device)], dim=1)
        # shifted_decoder_attention_mask = torch.cat([decoder_attention_mask[:, 1:], torch.zeros((decoder_attention_mask.size(0), 1), device=device, dtype=torch.long)], dim=1)

        # Calculate loss with mask
        logits_flat = outputs['logits'].view(-1, outputs['logits'].size(-1))
        labels_flat = shifted_labels.repeat_interleave(repeats=args.n_docs, dim=0).view(-1)
        # mask_flat = shifted_decoder_attention_mask.repeat_interleave(repeats=args.n_docs, dim=0).view(-1)
        loss = self.criterian(logits_flat, labels_flat)
        # loss = (loss * mask_flat).sum() / mask_flat.sum()

        outputs['loss'] = loss

        return outputs

training_args = Seq2SeqTrainingArguments(
    output_dir=args.model_saving_path,
    evaluation_strategy="steps",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    # num_train_epochs=1,
    save_safetensors=False,
    max_steps=args.max_steps,
    do_eval=True,
    save_steps=args.save_steps,
    logging_steps=args.logging_steps,
    eval_steps=args.eval_steps,
)

rag = RAG(model)

trainer = Seq2SeqTrainer(
    model=rag,
    args=training_args,
    train_dataset=encoded_tds,
    eval_dataset=encoded_vds,
    tokenizer=encoder_tokenizer,
)

trainer.train()

rag.model.question_encoder.save_pretrained(args.model_saving_path + "/encoder")
encoder_tokenizer.save_pretrained(args.model_saving_path + "/encoder")
rag.model.generator.save_pretrained(args.model_saving_path + "/generator")
generator_tokenizer.save_pretrained(args.model_saving_path + "/generator")
rag.model.retriever.save_pretrained(args.model_saving_path + "/retriever")

querys = [
    "원자력 발전소 건설 설계 단계에서 참고해야 하는 법률의 종류",
    "유사수신행위를 금지·처벌하는 유사수신행위의 규제에 관한 법률 제6조 제1항, 제3조 위반죄가 사기죄와 별개의 범죄인지 여부(적극)",
    "입양의 의사로 친생자 출생신고를 하고 입양의 실질적 요건이 모두 구비된 경우, 입양의 효력이 발생하는지 여부(적극)"
]

for query in querys:
    # generate
    prompt = f"다음 내용을 보고 참고해야 할 법률을 찾아주세요. {query}"
    inputs = encoder_tokenizer(prompt, return_tensors="pt").to(device)

    # Encode
    question_hidden_states = rag.model.question_encoder(inputs["input_ids"], inputs["attention_mask"])[0].mean(1).cpu()
    # Retrieve
    docs_dict = rag.model.retriever(inputs["input_ids"].cpu().numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
    # print(encoder_tokenizer.decode(docs_dict['context_input_ids'][0]))
    doc_scores = torch.bmm(
        question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
    ).squeeze(1)
    # Forward to generator
    generated_seq = rag.model.generate(
        context_input_ids=docs_dict["context_input_ids"].to(device),
        context_attention_mask=docs_dict["context_attention_mask"].to(device),
        doc_scores=doc_scores.to(device),
        do_sample=True,
        max_length=200,
        temperature=0.8,
    )
    
    print("query :", query)
    print("output :", generator_tokenizer.decode(generated_seq[0]) + '\n')