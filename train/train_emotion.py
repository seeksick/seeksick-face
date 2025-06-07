import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
from models.emotion_model import EmotionModel

# ====================
# 1. 하이퍼파라미터 및 설정
# ====================
DATA_PATH = "data/train"
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================
# 2. 전처리 및 데이터 로더 (5개 감정 필터링) [ 행복, 우울, 놀람, 분노, 중립 ]
# ====================
selected_classes = ['happy', 'sad', 'surprise', 'angry', 'neutral']
class_name_map = {
    'happy': '행복',
    'sad': '우울',
    'surprise': '놀람',
    'angry': '분노',
    'neutral': '중립'
}
class_to_new_index = {cls: i for i, cls in enumerate(selected_classes)}

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

all_dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)

selected_indices = [i for i, (_, label) in enumerate(all_dataset.samples)
                    if all_dataset.classes[label] in selected_classes]

class RemappedSubset(Dataset):
    def __init__(self, base_dataset, indices, class_map):
        self.base_dataset = base_dataset
        self.indices = indices
        self.class_map = class_map
        self.class_to_idx = class_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, original_label = self.base_dataset[real_idx]
        class_name = self.base_dataset.classes[original_label]
        new_label = self.class_map[class_name]
        return image, new_label

train_dataset = RemappedSubset(all_dataset, selected_indices, class_to_new_index)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ====================
# 3. 모델 및 학습 설정
# ====================
model = EmotionModel(num_emotions=5).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ====================
# 4. 학습 루프
# ====================
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 미니배치 단위 출력 (선택)
        # print(f"  [Epoch {epoch+1}][Batch {batch_idx+1}/{len(train_loader)}] Batch Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f">>> Epoch [{epoch+1}/{EPOCHS}] 완료 - 평균 Loss: {avg_loss:.4f}")

# ====================
# 5. 모델 저장
# ====================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"checkpoints/emotion_resnet18_5class_{timestamp}.pth"

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"[INFO] 모델 저장 완료: {save_path}")

# $env:PYTHONPATH="."; python train/train_emotion.py