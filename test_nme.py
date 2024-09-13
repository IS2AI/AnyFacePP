import os
import argparse
import cv2
from ultralytics import YOLO
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error
model = YOLO("model/best-maug.pt")
import matplotlib.pyplot as plt
import seaborn as sns
import math
#dataloader = dataloader or get_dataloader(data.get("test"), 1)
#metrics=model.val(data="exp.yaml",split="test")
#model.eval()
def get_image_dimensions(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Get image dimensions
    height, width, _ = image.shape

    return width, height

def denormalize_landmarks(landmarks, image_width, image_height):
    """
    Denormalize landmarks.

    Args:
    - landmarks (list): List of normalized landmarks [x1, y1, x2, y2, ..., xn, yn].
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.

    Returns:
    - list: Denormalized landmarks [x1, y1, x2, y2, ..., xn, yn].
    """
    denormalized_landmarks = []

    for i in range(0, len(landmarks)):
        x_normalized, y_normalized = landmarks[i][0], landmarks[i][1]

        # Denormalize the coordinates
        x_denormalized = x_normalized * image_width
        y_denormalized = y_normalized * image_height

        denormalized_landmarks.append([x_denormalized, y_denormalized])

    return denormalized_landmarks

def match(ground_truth_bboxes,predicted_bboxes):
        # Set a distance threshold for matching
    distance_threshold = 20

    # Initialize an array to store matched pairs (index in predicted, index in ground truth)
    sort_gt = []
    sort_pred=[]
    # Iterate through predicted landmarks
    for i,pred in enumerate(predicted_bboxes):
        # Find the corresponding bounding box for the predicted landmark
        pred_bbox = predicted_bboxes[i]

        # Iterate through ground truth bounding boxes
        for j, gt_bbox in enumerate(ground_truth_bboxes):
            aa=gt_bbox[0]
            # Calculate the distance between the centroids
            distance = np.linalg.norm(np.array(pred_bbox[0]) - np.array(gt_bbox[0]))

            # Check if the distance is below the threshold
            if distance < distance_threshold:
                # Add the pair to the matched pairs list
                sort_gt.append(gt_bbox[0:])
                sort_pred.append(pred_bbox[2:])
                break
    return sort_gt,sort_pred


def calculate_nme(ground_truth_landmarks, predicted_landmarks, eye_distances):
    """
    Calculate Normalized Mean Error (NME) for facial landmark detection.

    Parameters:
    - ground_truth_landmarks: List of arrays representing ground truth facial landmarks.
    - predicted_landmarks: List of arrays representing predicted facial landmarks.
    - eye_distances: List of eye distances for each detected face.

    Returns:
    - nme: Normalized Mean Error.
    """
    num_faces = len(ground_truth_landmarks)
    num_landmarks = 5
    calc_faces=0
    total_error = 0.0
    land_err=0
    print("nme",num_faces)
    for i in range(num_faces):
        print("hey",ground_truth_landmarks[i][0])
        if ground_truth_landmarks[i][0][0]<=-1:
            print("hey")
            continue
        if i<len(predicted_landmarks):
            for j in range(num_landmarks):
                # Calculate L2 norm between ground truth and predicted landmarks
                landmark_error = np.linalg.norm(predicted_landmarks[i][j][0] - float(ground_truth_landmarks[i][j][0]))+np.linalg.norm(predicted_landmarks[i][j][1] - float(ground_truth_landmarks[i][j][1]))
                land_err+=landmark_error
                # Normalize by the number of landmarks and eye distance
        if eye_distances[i]==0.0:continue
        normalized_error = landmark_error / (num_landmarks * eye_distances[i])
        print("here",normalized_error)
            
        total_error += normalized_error
        calc_faces=calc_faces+1

    # Calculate average NME across all landmarks and faces
    nme = total_error / num_faces
    if math.isinf(nme):
        print("here")

    return nme,calc_faces

def test(image,show):
    pred_fl=[]
    gt_FL=[]

    labels_path=image.replace("/images/","/labels/")
    with open(labels_path.split(".")[0]+".txt", "r") as file:
        gt_labels = file.readlines()
        image_width, image_height = get_image_dimensions(image)
        for line in gt_labels:
            labels=line.replace("\n","").split(" ")
            labels=np.array(labels[1:15])
            gt_fl=labels.reshape(7,2).astype(float)
            
            gt_fl = denormalize_landmarks(gt_fl,image_width, image_height)
            gt_FL.append(gt_fl)



    results = model.predict(source=image, imgsz=640)
    result = results[0].cpu().numpy()
    box=result.boxes.boxes
    box=box[:, 0:4]
    
    landmarks=[]
    d=len(box)
    for i in range(len(box)):
        dd=box[i]
        if len(box)!=0:
            kpt=result.keypoints.data
            kp_x1, kp_y1, kp_x2, kp_y2,kp_x3, kp_y3,kp_x4, kp_y4,kp_x5, kp_y5 = kpt[i][0][0],kpt[i][0][1],kpt[i][1][0],kpt[i][1][1],kpt[i][2][0],kpt[i][2][1],kpt[i][3][0],kpt[i][3][1],kpt[i][4][0],kpt[i][4][1]
            width = (box[i][2] - box[i][0])
            height = (box[i][3] - box[i][1])            
            x_center = (box[i][0] + width/2)
            y_center = (box[i][1] + height/2)

            landmarks.append([[x_center,y_center],[width,height],[kp_x1, kp_y1],[ kp_x2, kp_y2], [kp_x3, kp_y3], [kp_x4, kp_y4], [kp_x5, kp_y5]])


    return landmarks,gt_FL

num=0
num_empty=0
dir='datasets/dataset/wider/val/images/'
pred_age=[]
pred_gen=[]
pred_em=[]
em_gt=[]
gen_gt=[]
age_gt=[]
nme_result=[]


for i in os.listdir(dir):
    num=num+1
    print(num)
    pred_fl,gt_fl=test(dir+i,show=True)
    norml=[]
    sort_gt,sort_pred=match(gt_fl,pred_fl)

    for k in range(len(sort_gt)):
        norml.append(np.sqrt(sort_gt[k][1][0]*sort_gt[k][1][1]))
    sort_gt_f=  [[sublist for sublist in row[2:]] for row in sort_gt]
    if len(sort_gt_f)==0:
        num_empty=num_empty+1
        continue
    nme,calc_faces=calculate_nme(sort_gt_f, sort_pred, norml)
    if calc_faces==0:
        num_empty=num_empty+1
        continue
    nme_result.append(nme)

# Calculate the normalized values
normalized_nme = np.array(nme_result) * 100  # Multiply by 100 to get percentage
nme_thresholds = np.linspace(0, 10, 100)

# Sort NME values
sorted_nme = np.sort(normalized_nme)
# Create an array representing the number of images
num_images = np.arange(1, len(nme_result) + 1)
cumulative_percentage = [np.sum(sorted_nme <= threshold) / len(nme_result) * 100 for threshold in nme_thresholds]
# Plot the scatter plot
plt.plot(nme_thresholds, cumulative_percentage, color='b', label='CED Curve')
plt.title('Cumulative Error Distribution (CED) Curve')
plt.xlabel('NME Normalized by Bounding Box Size (%)')
plt.ylabel('Cumulative Percentage of Images (%)')
plt.legend()
plt.savefig('ced_curve.png')

# Print the result
print("NME:", np.sum(nme_result)/len(nme_result))





