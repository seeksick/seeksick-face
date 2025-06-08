# seeksick-face

```sh
emotion_recognition/
├── data/
│   └── # 감정 인식 학습용 데이터셋
├── checkpoints/
│   └── # 학습된 모델 저장 위치
├── models/
│   └── # ResNet18 기반 모델 구조 정의
├── train/
│   └── # 감정 인식 모델 학습 파이프라인
├── inference/
│   └── # 실시간 웹캠 감정 추론 코드
├── utils/
│
├── .gitignore       # git ignore
├── requirements.txt # 설치할 패키지 목록
└── README.md        # 프로젝트 설명 및 실행 방법

```

```sh
1. clone repository
$ git clone https://github.com/seeksick/seeksick-face.git
$ cd seeksick-face

2. 가상환경 생성 및 활성화
(for linux)
$ python3 -m venv .venv
$ source .venv/bin/activate

(for win)
$ .\micromamba.exe create -n faceenv python=3.10
$ .\micromamba.exe activate faceenv
$ micromamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
$ micromamba install opencv numpy pillow python-dotenv pandas -c conda-forge

3. install required libraries
$ pip install -r requirements.txt

4. 학습 실행
$ $env:PYTHONPATH="."; python train/train_emotion.py

5. 테스트 실행
$ $env:PYTHONPATH="."; python test/test_emotion.py

7. 학습 결과
- checkpoints/ 폴더에 .pth 파일이 생성됨
- test 실행 시 감정별 확률 출력 + 정확도 / 손실값 표시

```
