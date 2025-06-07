import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from models.emotion_model import EmotionModel

# ====================
# 1. 설정
# ====================
DATA_PATH = "data/test"
CHECKPOINT_PATH = "checkpoints/emotion_resnet18_5class_20250607_153158.pth"  # 최신 모델 경로로 수정
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================
# 2. 감정 클래스 (5개 감정 기준)
# ====================
selected_classes = ['happy', 'sad', 'surprise', 'angry', 'neutral']
class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}

# ====================
# 3. 전처리 및 데이터셋
# ====================
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

all_dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)

# 선택 클래스만 필터링
selected_indices = [i for i, (_, label) in enumerate(all_dataset.samples)
                    if all_dataset.classes[label] in selected_classes]

class RemappedSubset(torch.utils.data.Dataset):
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

test_dataset = RemappedSubset(all_dataset, selected_indices, class_to_idx)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ====================
# 4. 모델 불러오기
# ====================
model = EmotionModel(num_emotions=5).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

criterion = nn.CrossEntropyLoss()

# ====================
# 5. 평가
# ====================
test_loss = 0.0
correct = 0
total = 0

idx_to_class = {v: k for k, v in class_to_idx.items()}
emotion_kor_map = {
    'happy': '행복',
    'sad': '우울',
    'surprise': '놀람',
    'angry': '분노',
    'neutral': '중립'
}

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # 예측 클래스
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # 예측 확률 벡터 출력
        probs = torch.softmax(outputs, dim=1).cpu()
        labels = labels.cpu()
        for i in range(images.size(0)):
            folder_name = idx_to_class[labels[i].item()]
            prob_vec = probs[i].tolist()
            prob_str = ', '.join(
                [f"{emotion_kor_map[selected_classes[j]]}: {prob_vec[j]:.3f}" for j in range(5)]
            )
            print(f"[{emotion_kor_map[folder_name]}] [{prob_str}]")

# 최종 결과
avg_loss = test_loss / len(test_loader)
accuracy = correct / total * 100

print(f"\n[Test] Accuracy: {accuracy:.2f}% | Loss: {avg_loss:.4f}")