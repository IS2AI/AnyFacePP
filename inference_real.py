from ultralytics import YOLO
import cv2

path_to_model="last.pt"
model = YOLO(path_to_model)
results=model.predict(source="0",show=True,iou=0.2,conf=0.6, device="cuda")
print("hello")