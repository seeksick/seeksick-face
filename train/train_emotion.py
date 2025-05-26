# train_emotion.py: 예제 데이터 기반 감정 벡터 회귀 학습 코드
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from models.emotion_model import EmotionModel

# ====================
# 1. 하이퍼파라미터 및 설정
# ====================
DATA_PATH = "data/emotion_labels.csv"
IMAGE_FOLDER = "data/images"
EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================
# 2. 데이터셋 정의
# ====================
class EmotionDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.data.iloc[idx]['image'])
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(eval(self.data.iloc[idx]['label']), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# ====================
# 3. 전처리 및 데이터 로더
# ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = EmotionDataset(DATA_PATH, IMAGE_FOLDER, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ====================
# 4. 모델 및 학습 설정
# ====================
model = EmotionModel(num_emotions=5).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ====================
# 5. 학습 루프
# ====================
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(dataloader):.4f}")

# ====================
# 6. 모델 저장
# ====================
torch.save(model.state_dict(), "checkpoints/emotion_resnet18.pth")
print("[INFO] 모델 저장 완료")