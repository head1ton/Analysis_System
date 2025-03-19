from sklearn.utils import compute_class_weight
import numpy as np
import evaluate

# 평가 지표로 정확도 로드
metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    # 평가 예측값과 라벨 추출
    logits, labels = eval_pred
    # 예측값 계산
    predictions = np.argmax(logits, axis=1)
    # 정확도 계산
    return metric.compute(predictions=predictions, references=labels)

def get_class_weights(df):
    # 클래스 가중치 계산
    class_weights = compute_class_weight("balanced",
                                         classes=sorted(df['label'].unique().tolist()),
                                         y=df['label'].tolist())
    return class_weights