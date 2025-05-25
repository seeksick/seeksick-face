import torch
import torch.nn as nn
from torchvision import models

class EmotionModel(nn.Module):
    def __init__(self, num_emotions=7):
        super(EmotionModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotions)  # 감정 벡터 출력
        )

    def forward(self, x):
        return self.resnet(x)