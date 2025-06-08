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
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ====================
# 4. 얼굴 검출기 및 웹캠 실행
# ====================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
print("[INFO] 웹캠이 실행되었습니다. 'q'를 눌러 종료하세요.")

try:
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- 전체 프레임에 격자 오버레이 ---
            grid_step = 28
            for i in range(0, frame.shape[1], grid_step):  # 세로줄
                cv2.line(frame, (i, 0), (i, frame.shape[0]), (100, 100, 100), 1)
            for i in range(0, frame.shape[0], grid_step):  # 가로줄
                cv2.line(frame, (0, i), (frame.shape[1], i), (100, 100, 100), 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                resized_face = cv2.resize(face, (224, 224))

                # 감정 추론
                input_tensor = preprocess(resized_face).unsqueeze(0).to(DEVICE)
                logits = model(input_tensor)[0].cpu()
                probs = F.softmax(logits, dim=0).numpy()

                # 감정 벡터 출력
                print(f"[{time.strftime('%H:%M:%S')}] 감정 벡터:", end=" ")
                for i, score in enumerate(probs):
                    print(f"{emotion_labels[i]}: {score:.2f}", end=" ")
                print()

                # 감정 시각화
                label_idx = np.argmax(probs)
                label = emotion_labels[label_idx]
                confidence = probs[label_idx]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # 얼굴 별도 보기
                cv2.imshow("Face 224x224", resized_face)

            cv2.imshow("Emotion Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 종료 완료")