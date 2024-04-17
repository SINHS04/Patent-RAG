import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
import argparse
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel

# get args
parser = argparse.ArgumentParser(description="make docs for retrieve")
parser.add_argument("--path", type=str, default="../data/preprocessed_law_context_hang.jsonl", help="path to preprocessed data")
parser.add_argument("--output_path", type=str, default="../data/RAG_document", help="output path to save dataset")
args = parser.parse_args()

path = args.path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device : {device}")

# get encoder (same as the one used in RAG_train.py)
encoder_id = "sentence-transformers/xlm-r-base-en-ko-nli-ststb"
encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_id)
encoder = AutoModel.from_pretrained(encoder_id)

encoder.eval()

# for embedding
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# read file
def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()

# get embeddings of docs
def get_embedding(lines, batch_size=8):
    titles = []
    texts = []
    embeddings = []
    batch = []
    for line in tqdm(lines):
        with torch.no_grad():
            data = json.loads(line)
            titles.append(data['title'])
            texts.append(data['text'])
            batch.append('<cls>' + data['title'] + ' / ' + data['text'])
            if len(batch) == batch_size:
                encoded_input = encoder_tokenizer(batch, padding=True, truncation=True, return_tensors='pt')

                with torch.no_grad():
                    model_output = encoder(**encoded_input)

                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                embeddings.extend(sentence_embeddings)
                batch = []

    if len(batch) > 0:
        encoded_input = encoder_tokenizer(batch, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = encoder(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings.extend(sentence_embeddings)
        batch = []

    return titles, texts, embeddings

raw_data = read_file(path)
titles, texts, embeddings = get_embedding(raw_data)

# save docs
docs = Dataset.from_dict({
    'title': titles, 
    'text': texts, 
    'embeddings': embeddings
})

docs.save_to_disk(args.output_path)
docs.add_faiss_index(column='embeddings')
docs.get_index("embeddings").save(args.output_path + ".faiss")