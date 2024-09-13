import os
os.environ["OMP_NUM_THREADS"]='2'

from ultralytics import YOLO
# Load a model
model = YOLO('yolov8m-pose.yaml')  # build a new model from YAML

# Train the model
model.train(data='exp.yaml', epochs=300, imgsz=640, batch=1, device="cpu")
