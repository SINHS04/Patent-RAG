# 법률 검색 모델
주어진 입력과 관련된 법을 검색하고, 이를 바탕으로 참조해야 하는 법을 생성해주는 모델입니다.

### Baseline
|Type|Model(huggingface)|
|:---:|:---:|
|question_encoder|sentence-transformers/xlm-r-base-en-ko-nli-ststb|
|generator|facebook/mbart-large-50|

## Directory Structue
```
# build docs(faiss)
build_docs
└── build_docs.py

# dataset
data
├── law_train_data.jsonl
└── law_valid_data.jsonl

# train and generate
model
├── RAG_train.py
└── RAG_generate.py

# dependency
requirements.txt
```

## Data Format
```
{
    "id": 23, 
    "context": "오피스텔이 재산세 과세대상인 ‘주택’에 해당하는지 판단하는 기준", 
    "target": [
        "지방세법 제6조 제4호", 
        "지방세법 제104조 제2호", 
        "지방세법 제104조 제3호", 
        "지방세법 제105조", 
        "건축법 제2조 제1항 제2호", 
        "건축법 제2항", 
        "건축법 시행령 제3조의5 [별표 1] 제14호"
    ]
}
```


## Enviroments
Install Python Dependency
```
pip install -r requirements.txt
```

## How to Run
### Train
```
python -m run train \
    --output-dir outputs/ \
    --seed 42 --epoch 10 \
    --learning-rate 2e-5 --weight-decay 0.01 \
    --batch-size 64 --valid-batch-size 64
```
python -m run train --output-dir outputs/beomi/KcELECTRA-base-v2022/1_2 --seed 42 --epoch 10 --learning-rate 2e-5 --weight-decay 0.01 --batch-size 64 --valid-batch-size 64

python -m run train --output-dir outputs --seed 42 --epoch 10 --learning-rate 2e-5 --weight-decay 0.01 --batch-size 64 --valid-batch-size 64 --model-num 2 --tokenizer-num 2 --model2-num 2 --tokenizer2-num 2

python -m run train --output-dir outputs/inputchange --seed 42 --epoch 10 --learning-rate 2e-5 --weight-decay 0.01 --batch-size 64 --valid-batch-size 64 --model-num 2 --tokenizer-num 2 --model2-num 2 --tokenizer2-num 2

### Inference
```
python -m run inference \
    --model-ckpt-path /workspace/Korean_EA_2023/outputs/<your-model-ckpt-path> \
    --output-path test_output.jsonl \
    --batch-size 64 \
    --device cuda:0
```
python -m run inference --model-ckpt-path outputs/klue/roberta-large/1_3/model --output-path outputs/klue/roberta-large/1_3/output.jsonl --batch-size 64 --device cuda:0

now
python -m run inference --model-ckpt-path outputs/nlp04/korean_sentiment_analysis_kcelectra/dataset_0/model --output-path outputs/nlp04/korean_sentiment_analysis_kcelectra/dataset_0/output.jsonl --batch-size 64 --device cuda:0

python -m run inference --model-ckpt-path outputs/nlp04/korean_sentiment_analysis_kcelectra/dataset-0_1/model --model2-ckpt-path outputs/nlp04/korean_sentiment_analysis_kcelectra/dataset-0_1/model2 --output-path outputs/nlp04/korean_sentiment_analysis_kcelectra/dataset-0_1/output.jsonl --batch-size 64 --device cuda:0 --device2 cuda:1

python -m run inference --model-ckpt-path outputs/model --model2-ckpt-path outputs/model2 --output-path outputs/output.jsonl --batch-size 64 --device cuda:0 --device2 cuda:1

