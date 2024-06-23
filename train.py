import os
os.environ["OMP_NUM_THREADS"]='2'
import torch

from ultralytics import YOLO
# Load a model
model = YOLO('last.pt')  # build a new model from YAML
# Train the model
model.train(data='dataset.yaml', epochs=200, imgsz=640, batch=36, device=[2,1], resume=True)
