import torch
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 입력 데이터에서 라벨 추출
        labels = inputs.get("labels")

        # 모델을 사용해서 출력 계산
        outputs = model(**inputs)
        logits = outputs.get("logits")
        logits = logits.float()

        # 클래스 가중치를 사용하여 손실 함수 설정
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights, dtype=torch.float).to(device=self.device))
        # 손실 계산
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def set_class_weights(self, class_weights):
        # 클래스 가중치 설정
        self.class_weights = class_weights

    def set_device(self, device):
        # 디바이스 설정
        self.device = device
