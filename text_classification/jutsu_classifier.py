import pandas as pd
import torch
import huggingface_hub
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          DataCollatorWithPadding,
                          TrainingArguments,
                          pipeline)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datasets import Dataset
import gc
from .cleaner import Cleaner

from .training_utils import get_class_weights, compute_metrics
from .custom_trainer import CustomTrainer

class JutsuClassifier():
    def __init__(self,
        model_path,
        data_path=None,
        text_column_name='text',
        label_column_name='jutsu',
        model_name="distilbert/distilbert-base-uncased",
        test_size=0.2,
        num_labels=3,
        huggingface_token=None):

        # 모델 경로, 데이터 경로, 텍스트 컬럼 이름, 라벨 컬럼 이름, 모델 이름, 테스트 데이터 비율, 라벨 수, 허깅페이스 토큰
        self.model_path = model_path
        self.data_path = data_path
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.model_name = model_name
        self.test_size = test_size
        self.num_labels = num_labels
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.huggingface_token = huggingface_token
        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)

        # 토크나이저 로드
        self.tokenizer = self.load_tokenizer()

        # 모델 경로가 허깅페이스 허브에 없으면 데이터 경로를 사용하여 모델을 학습
        if not huggingface_hub.repo_exists(self.model_path):
            if data_path is None:
                raise ValueError("Data path is required to train the model, since the model path does not exist in huggingface hub")

            # 데이터 로드
            train_data, test_data = self.load_data(self.data_path)
            train_data_df = train_data.to_pandas()
            test_data_df = test_data.to_pandas()

            # 모든 데이터를 합쳐서 클래스 가중치 계산
            all_data = pd.concat([train_data_df, test_data_df]).reset_index(drop=True)
            class_weights = get_class_weights(all_data)

            # 모델 학습
            self.train_model(train_data, test_data, class_weights)

        # 모델 로드
        self.model = self.load_model(self.model_path)

    def load_model(self, model_path):
        # 텍스트 분류 파이프라인을 사용하여 모델 로드
        model = pipeline('text-classification', model=model_path, return_all_scores=True)
        return model

    def train_model(self, train_data, test_data, class_weights):
        # 사전 학습된 모델 로드
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                                   num_labels=self.num_labels,
                                                                   id2label=self.label_dict,)
        # 데아터 콜레이터 설정
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # 학습 인자 설정
        training_args = TrainingArguments(
            output_dir=self.model_path,
            learning_rate=2e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            push_to_hub=True,
        )
        # 커스텀 트레이너 설정
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # 디바이스와 클래스 가충치 설정
        trainer.set_device(self.device)
        trainer.set_class_weights(class_weights)

        # 모델 학습
        trainer.train()

        # 메모리 정리
        del trainer, model
        gc.collect()

        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def simplify_jutsu(self, jutsu):
        if "Genjutsu" in jutsu:
            return "Genjutsu"
        if "Ninjutsu" in jutsu:
            return "Ninjutsu"
        if "Taijutsu" in jutsu:
            return "Taijutsu"

    def preprocess_function(self, tokenizer, examples):
        # 텍스트를 토크나이저로 전처리
        return tokenizer(examples['text_cleaned'], truncation=True)

    def load_data(self, data_path):

        df = pd.read_json(data_path, lines=True)
        df['jutsu_type_simplified'] = df['jutsu_type'].apply(self.simplify_jutsu)
        df['text'] = df['jutsu_name'] + ". " + df['jutsu_description']
        df[self.label_column_name] = df['jutsu_type_simplified']
        df = df[['text', self.label_column_name]]
        df = df.dropna()

        cleaner = Cleaner()
        df['texts_cleaner'] = df[self.text_column_name].apply(cleaner.clean)

        # 라벨 인코딩
        le = preprocessing.LabelEncoder()
        le.fit(df[self.label_column_name].tolist())

        # 라벨 딕셔너리 생성
        label_dict = {index:label_name for index, label_name in enumerate(le.__dict__['classes_'].tolist())}
        self.label_dict = label_dict
        df['label'] = le.transform(df[self.label_column_name].tolist())

        # 데이터 분할
        test_size = 0.2
        df_train, df_test = train_test_split(df, test_size=test_size, stratify=df['label'])

        # 데이터셋 생성
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)

        # 데아터 전처리
        tokenized_train = train_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples), batched=True)
        tokenized_test = test_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples), batched=True)

        return tokenized_train, tokenized_test

    def load_tokenizer(self):
        # 토크나이저 로드
        if huggingface_hub.repo_exists(self.model_path):
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer

    def postprocess(self, model_output):
        # 모델 출력 후처리
        output = []
        for pred in model_output:
            label = max(pred, key=lambda x: x['score'])['label']
            output.append(label)
        return output

    def classify_jutsu(self, text):
        model_output = self.model(text)
        predictions = self.postprocess(model_output)
        return predictions