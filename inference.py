import cv2
from ultralytics import YOLO
import numpy as np
import requests
import json

# Define the model and image paths
path_to_model = "best-maug.pt"
path_to_image = "example.jpg"
image_url = None
threshold_bboxes = 0.3
model = YOLO(path_to_model)
iou = 0.1

def get_image_dimensions(image_path):
    """
    Get image dimensions using OpenCV.
    
    Args:
    - image_path (str): Path to the image file.

    Returns:
    - tuple: Width and height of the image.
    """
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    return width, height

def denormalize_landmarks(landmarks, image_width, image_height):
    """
    Denormalize facial landmarks to the original image size.

    Args:
    - landmarks (list): Normalized facial landmarks [(x1, y1), (x2, y2), ...].
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.

    Returns:
    - list: Denormalized landmarks.
    """
    return [[x * image_width, y * image_height] for x, y in landmarks]

def test(image_path, image_url, threshold_bboxes, iou, show, save_predictions=False):
    """
    Run face detection and attribute prediction using the YOLO model.

    Args:
    - image_path (str): Path to the input image.
    - image_url (str): URL to fetch the image (optional).
    - threshold_bboxes (float): Confidence threshold for bounding boxes.
    - iou (float): Intersection over Union (IoU) threshold.
    - show (bool): Whether to display the output image.
    - save_predictions (bool): Whether to save the predictions to a JSON file.
    """
    # Load image from URL or path
    if image_url:
        response = requests.get(image_url)
        image_array = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_path)

    # Run the YOLO model for predictions
    results = model.predict(source=image, imgsz=640, conf=threshold_bboxes, iou=iou, device='cuda')
    result = results[0].cpu().numpy()
    boxes = result.boxes.boxes
    H, W, _ = image.shape
    font_size = 0.7

    face_predictions = []

    for i in range(len(boxes)):
        if boxes.size != 0:
            # Extract bounding box and label information
            x1, y1, x2, y2, confidence, label = map(int, boxes[i][:6])
            width = x2 - x1
            height = y2 - y1

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), int((W+H)/640)+1)

            # Determine face type label
            label_map = {0: 'human', 1: 'animal', 2: 'cartoon'}
            label_text = label_map.get(label, 'unknown')
            mtl=results[0].mtl
            # Extract gender, age, and emotion predictions
            if label_text=='human':
                age=str(int(mtl[i][3:4][0]))
            else:
                age="unsure"
            
            pred_GEN=mtl[i][0:3]
            class_GEN = np.argmax(pred_GEN.cpu())
            class_labels_GEN = ['female', 'male', 'unsure']
            predicted_class_GEN = class_labels_GEN[class_GEN]

            
            pred_EM=mtl[i][4:]
            class_EM = np.argmax(pred_EM.cpu())
            emotion_text = ['angry', 'happy', 'fear', 'sad', 'surprise', 'disgust', 'neutral','unsure'][class_EM]

            # Display text information on the image
            cv2.putText(image, f'Face: {label_text}', (x1+2, y1 - int(H/8)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 1)
            cv2.putText(image, f'Gender: {predicted_class_GEN}', (x1+2, y1 - int(H/15)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 0), 1)
            cv2.putText(image, f'Age: {age}', (x1+2, y1 - int(H/10)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 255), 1)
            cv2.putText(image, f'Emotion: {emotion_text}', (x1+2, y1 - int(H/25)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), 1)

            # Draw landmarks
            keypoints = results[0].keypoints.data[i]
            landmarks = [(float(kp[0]), float(kp[1])) for kp in keypoints]
            landmark_colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 0, 0)]
            for k, (x, y) in enumerate(landmarks):
                cv2.circle(image, (int(x), int(y)), 2, landmark_colors[k % len(landmark_colors)], -1)

            # Create face prediction dictionary
            face_prediction = {
                "label": label_text,
                "confidence": int(confidence * 100),
                "bounding_box": {"x": x1, "y": y1, "width": width, "height": height},
                "gender": predicted_class_GEN,
                "age": age,
                "emotion": emotion_text,
                "facial_landmarks": landmarks
            }
            face_predictions.append(face_prediction)

    # Save predictions to a JSON file if save_predictions is True
    if save_predictions:
        with open("predictions.json", "w") as json_file:
            json.dump(face_predictions, json_file, indent=2)

    # Optionally display the result image
    if show:
        cv2.imwrite('out.png', image)
        cv2.imshow('Image with Bounding Box', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Run the test function with specified parameters
test(path_to_image, image_url, threshold_bboxes, iou, show=True, save_predictions=True)
