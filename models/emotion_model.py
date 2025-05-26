import torch
import torch.nn as nn
from torchvision import models

class EmotionModel(nn.Module):
    def __init__(self, num_emotions=5):  # 감정 5개
        super(EmotionModel, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotions),  # 5차원 감정 강도 출력
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)