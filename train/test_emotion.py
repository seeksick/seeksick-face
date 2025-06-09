import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import csv
from sklearn.metrics import f1_score, classification_report
from models.emotion_model import EmotionModel

# ====================
# 1. 설정
# ====================
DATA_PATH = "data/test"
CHECKPOINT_PATH = "checkpoints/emotion_resnet18_5class_20250608_170626.pth"  # 최근 모델
CSV_PATH = "checkpoints/emotion_predictions_latest.csv"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================
# 2. 감정 클래스
# ====================
selected_classes = ['happy', 'sad', 'surprise', 'angry', 'neutral']
class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
idx_to_class = {v: k for k, v in class_to_idx.items()}
emotion_kor_map = {
    'happy': '행복',
    'sad': '우울',
    'surprise': '놀람',
    'angry': '분노',
    'neutral': '중립'
}

# ====================
# 3. 전처리 및 데이터셋
# ====================
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

all_dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
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
# 5. 평가 및 CSV 저장
# ====================
test_loss = 0.0
correct = 0
total = 0
csv_rows = []
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        probs = torch.softmax(outputs, dim=1).cpu()
        labels = labels.cpu()
        predicted = torch.argmax(probs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.tolist())
        all_labels.extend(labels.tolist())

        for i in range(images.size(0)):
            true_label_idx = labels[i].item()
            true_label_eng = idx_to_class[true_label_idx]
            true_label_kor = emotion_kor_map[true_label_eng]

            prob_vec = probs[i].tolist()
            row = {
                "TrueLabel": true_label_kor,
                "행복": round(prob_vec[0], 3),
                "우울": round(prob_vec[1], 3),
                "놀람": round(prob_vec[2], 3),
                "분노": round(prob_vec[3], 3),
                "중립": round(prob_vec[4], 3)
            }
            csv_rows.append(row)

# CSV 저장
os.makedirs("checkpoints", exist_ok=True)
with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["TrueLabel", "행복", "우울", "놀람", "분노", "중립"])
    writer.writeheader()
    writer.writerows(csv_rows)

# ====================
# 6. 최종 결과 출력
# ====================
avg_loss = test_loss / len(test_loader)
accuracy = correct / total * 100
macro_f1 = f1_score(all_labels, all_preds, average='macro')

print(f"\n[Test] Accuracy: {accuracy:.2f}% | Loss: {avg_loss:.4f} | Macro F1: {macro_f1:.4f}")
print(f"[INFO] 결과 CSV 저장 완료: {CSV_PATH}\n")

print("[클래스별 F1 스코어]")
print(classification_report(all_labels, all_preds, target_names=selected_classes, digits=3))