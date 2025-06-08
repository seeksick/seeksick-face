import cv2
import torch
import numpy as np
from torchvision import transforms
from models.emotion_model import EmotionModel
from dotenv import load_dotenv
import os
import time
import torch.nn.functional as F

# ====================
# 1. 설정
# ====================
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/emotion_resnet18.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_labels = ['happy', 'sad', 'surprise', 'angry', 'neutral']

# ====================
# 2. 모델 불러오기
# ====================
model = EmotionModel(num_emotions=5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# ====================
# 3. 전처리 (224x224, BICUBIC)
# ====================
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ====================
# 4. 웹캠 시작
# ====================
cap = cv2.VideoCapture(0)
print("[INFO] 웹캠이 실행되었습니다. 'q'를 눌러 종료하세요.")

try:
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- 타원 중심 및 마스크 생성 ---
            frame_copy = frame.copy()
            blurred = cv2.GaussianBlur(frame_copy, (31, 31), 0)
            mask = np.zeros_like(frame, dtype=np.uint8)
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            axes = (400, 350)  # 타원 크기 확대

            # 타원 마스크 및 반전
            cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, (255, 255, 255), -1)
            mask_inv = cv2.bitwise_not(mask)

            # 영역 분리
            face_area = cv2.bitwise_and(frame, mask)
            blurred_area = cv2.bitwise_and(blurred, mask_inv)
            combined = cv2.add(face_area, blurred_area)

            # 얼굴 분석 영역 잘라내기
            x1, y1 = center_x - axes[0], center_y - axes[1]
            x2, y2 = center_x + axes[0], center_y + axes[1]
            ellipse_roi = face_area[y1:y2, x1:x2]

            if ellipse_roi.shape[0] > 0 and ellipse_roi.shape[1] > 0:
                input_tensor = preprocess(ellipse_roi).unsqueeze(0).to(DEVICE)
                logits = model(input_tensor)[0].cpu()
                probs = F.softmax(logits, dim=0).numpy()

                # 콘솔 출력
                print(f"[{time.strftime('%H:%M:%S')}] 감정 벡터:", end=" ")
                for i, score in enumerate(probs):
                    print(f"{emotion_labels[i]}: {score:.2f}", end=" ")
                print()

                # 시각화 라벨 표시
                label_idx = np.argmax(probs)
                label = emotion_labels[label_idx]
                confidence = probs[label_idx]
                cv2.putText(combined, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # 흰색 타원 표시 (두께 3)
            cv2.ellipse(combined, (center_x, center_y), axes, 0, 0, 360, (255, 255, 255), 3)

            # 출력
            cv2.imshow("Emotion Detection", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 종료 완료")