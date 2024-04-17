import torch
import argparse
from transformers import (
  RagRetriever, 
  RagConfig,
  RagTokenForGeneration,
  AutoModelForSeq2SeqLM, 
  AutoTokenizer, 
  AutoModel, 
  MBartConfig, 
  XLMRobertaConfig
)

parser = argparse.ArgumentParser(description="generate")
parser.add_argument("--docs_dir", type=str, default="./data/RAG_document", help="docs directory")
parser.add_argument("--faiss_path", type=str, default="./data/RAG_document.faiss", help="docs faiss path")
parser.add_argument("--n_docs", type=int, default=3, help="number of docs to retrieve")
parser.add_argument("--model_saving_path", type=str, default="./trained_models/rag", help="model saving path")
args = parser.parse_args()

print("Model is loading...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder_id = args.model_saving_path + "/encoder"
encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_id)
encoder = AutoModel.from_pretrained(encoder_id).to(device)

generator_id = args.model_saving_path + "/generator"
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

print("Model is ready!")

for i in range(10):
    prompt = input(f"입력(남은 횟수 : {10-i}) : ")
    inputs = encoder_tokenizer(prompt, return_tensors="pt").to(device)

    # Encode
    question_hidden_states = model.question_encoder(inputs["input_ids"], inputs["attention_mask"])[0].mean(1).cpu()

    # Retrieve
    docs_dict = model.retriever(inputs["input_ids"].cpu().numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
    doc_scores = torch.bmm(
        question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
    ).squeeze(1)

    # Forward to generator
    generated_seq = model.generate(
        context_input_ids=docs_dict["context_input_ids"].to(device),
        context_attention_mask=docs_dict["context_attention_mask"].to(device),
        doc_scores=doc_scores.to(device),
        do_sample=True,
        max_length=200,
        temperature=0.8,
    )

    print(generator_tokenizer.decode(generated_seq[0]) + '\n')