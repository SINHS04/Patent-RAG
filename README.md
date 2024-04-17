# 법률 검색 모델
(Prototype)주어진 입력과 관련된 법을 검색하고, 이를 바탕으로 참조해야 하는 법을 생성해주는 모델입니다.

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
├── law_valid_data.jsonl
├── RAG_document.faiss
└── RAG_document
    ├── data-00000-of-00002.arrow
    ├── data-00000-of-00003.arrow
    ├── data-00001-of-00002.arrow
    ├── data-00001-of-00003.arrow
    ├── data-00002-of-00003.arrow
    ├── dataset_info.json
    └── state.json

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
python ./model/RAG_train.py
```

Options
```
options:
  -h, --help            show this help message and exit

  --docs_dir DOCS_DIR   
                        docs directory

  --faiss_path FAISS_PATH
                        docs faiss path

  --n_docs N_DOCS       
                        number of docs to retrieve

  --train_path TRAIN_PATH
                        train data path

  --valid_path VALID_PATH
                        valid data path

  --lr LR               
                        learning rate

  --batch_size BATCH_SIZE
                        batch size

  --max_steps MAX_STEPS
                        max steps

  --save_steps SAVE_STEPS
                        save steps

  --eval_steps EVAL_STEPS
                        eval steps

  --logging_steps LOGGING_STEPS
                        logging steps

  --model_saving_path MODEL_SAVING_PATH
                        model saving path
```

### Generate
```
python ./model/RAG_generate.py
````

Options
````
options:
  -h, --help            show this help message and exit

  --docs_dir DOCS_DIR   
                        docs directory

  --faiss_path FAISS_PATH
                        docs faiss path

  --n_docs N_DOCS       
                        number of docs to retrieve

  --model_saving_path MODEL_SAVING_PATH
                        model saving path
```

## Outptu example
```
입력(남은 횟수 : 10) : 원자력 발전소
</s><s> 원자력안전법 / 제1절 발전용원자로 및 관계시설의 건설 // 발전소</s>
```