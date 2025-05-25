import cv2
import torch
import numpy as np
from torchvision import transforms
from models.emotion_model import EmotionModel
from dotenv import load_dotenv
import os

# 환경 변수 로드 (.env 파일에서)
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/emotion_resnet18.pth")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# 감정 라벨 정의
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 모델 불러오기
model = EmotionModel(num_emotions=len(emotion_labels))
#model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# 전처리
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 얼굴 검출기: OpenCV haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 웹캠 시작
cap = cv2.VideoCapture(0)

print("[INFO] 웹캠이 실행되었습니다. 'q'를 눌러 종료하세요.")

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            input_tensor = preprocess(face).unsqueeze(0).to(DEVICE)

            output = model(input_tensor)

            # [1] 로그잇 벡터 출력 (softmax 전)
            # print("[Log ]", output[0].cpu().numpy())

            # [2] 소프트맥스 후 확률 벡터
            probs = torch.softmax(output[0], dim=0).cpu().numpy()

            # [3] 감정별 확률 출력
            for label, prob in zip(emotion_labels, probs):
                print(f"{label}: {prob:.2f}", end="  ")
            print()

            # [4] 가장 높은 확률 감정 추출
            label_idx = np.argmax(probs)
            label = emotion_labels[label_idx]
            confidence = probs[label_idx]

            # 시각화
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Emotion Detection (ResNet18)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()