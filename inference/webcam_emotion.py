import cv2
import torch
import numpy as np
from torchvision import transforms
from models.emotion_model import EmotionModel
from dotenv import load_dotenv
import os
import time

# 환경 변수 로드 (.env 파일에서)
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/emotion_resnet18.pth")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# 감정 라벨 정의 (5개 감정 강도)
emotion_labels = ['행복', '슬픔', '기쁨', '분노', '우울']

# 모델 불러오기
model = EmotionModel(num_emotions=len(emotion_labels))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # 실제 학습된 모델 있다면 주석 해제
model.to(DEVICE).eval()

# 전처리
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 얼굴 검출기: OpenCV haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 웹캠 시작
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] 웹캠이 실행되었습니다. 'q'를 눌러 종료하세요.")

# 감정 추출 주기 설정
timestamp = time.time()
prediction_interval = 0.5  # 초 단위 (0.5초마다 감정 추론)
label = ""
confidence = 0.0

try:
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            now = time.time()

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]

                if now - timestamp >= prediction_interval:
                    timestamp = now

                    input_tensor = preprocess(face).unsqueeze(0).to(DEVICE)
                    output = model(input_tensor)[0].cpu().numpy()  # 소프트맥스 없이 감정 벡터 그대로 사용

                    # [1] 감정 강도 벡터 출력
                    print(f"[{time.strftime('%H:%M:%S')}]-[감정 벡터] ", end="")
                    for label_text, score in zip(emotion_labels, output):
                        print(f"{label_text}: {score:.2f}", end=" ")
                    print()

                    # [2] 가장 강한 감정 선택 (선택적)
                    label_idx = np.argmax(output)
                    label = emotion_labels[label_idx]
                    confidence = output[label_idx]

                # 시각화
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow("Emotion Detection (Regression)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if cv2.getWindowProperty("Emotion Detection (Regression)", cv2.WND_PROP_VISIBLE) < 1:
                break

except KeyboardInterrupt:
    print("\n[INFO] 강제 종료됨")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 종료 완료")