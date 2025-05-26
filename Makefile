.PHONY: train

infer:
	PYTHONPATH=. python3 inference/webcam_emotion.py

train:
	PYTHONPATH=. python3 train/train_emotion.py