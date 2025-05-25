# seeksick-face

```python

emotion_recognition/
├── data/
│   └── .csv                  # (선택) 감정 인식 학습용 데이터셋
├── checkpoints/
│   └── .pth         # 학습된 모델 저장 위치
├── models/
│   └── emotion_model.py             # ResNet18 기반 모델 구조 정의
├── train/
│   └── train_emotion.py             # 감정 인식 모델 학습 파이프라인
├── inference/
│   └── webcam_emotion.py            # 실시간 웹캠 감정 추론 코드
├── utils/
│   ├── data_loader.py               # FER2013용 커스텀 Dataset 및 DataLoader
│   └── metrics.py                   # 정확도, loss 계산 등 지표 함수
├── requirements.txt                 # 설치할 패키지 목록
└── README.md                        # 프로젝트 설명 및 실행 방법

```