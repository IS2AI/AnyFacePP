import os
import argparse
import cv2
from ultralytics import YOLO
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error
model = YOLO("last.pt")
import matplotlib.pyplot as plt
import seaborn as sns

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
    distance_threshold = 50

    # Initialize an array to store matched pairs (index in predicted, index in ground truth)
    sort_gt = []
    sort_pred=[]
    index_pred=[]
    index_gt=[]
    
    # Iterate through predicted landmarks
    for j,gt_bbox in enumerate(ground_truth_bboxes):
        # Find the corresponding bounding box for the predicted landmark
        gt=[]
        sc=50
        # Iterate through ground truth bounding boxes
        for i, pred in enumerate(predicted_bboxes):
            pred_bbox = predicted_bboxes[i]

            # Calculate the distance between the centroids
            distance = np.linalg.norm(np.array(pred_bbox[0]) - np.array(gt_bbox[0]))
            print(distance)

            # Check if the distance is below the threshold
            if distance < distance_threshold:
                # Add the pair to the matched pairs list
                if sc>distance: 
                    gt=gt_bbox[0:]
                    predd=pred_bbox
                    index_i=i
                    index_j=j
                    sc=distance
        if len(gt)>0:
            sort_gt.append(gt)
            sort_pred.append(predd)
            index_pred.append(index_i)
            index_gt.append(index_j)
          
    return sort_gt,sort_pred,index_gt,index_pred


def test(image,show):
    pred_age_ev=[]
    pred_em_ev=[]
    pred_gen_ev=[]
    gt_age=[]
    gt_em=[]
    gt_gen=[]
    gt_bbs=[]
    labels_path=image.replace("/images/","/labels/")
    with open(labels_path.split(".")[0]+".txt", "r") as file:
        gt_labels = file.readlines()
        image_width, image_height = get_image_dimensions(image)
        for line in gt_labels:
            labels=line.replace("\n","").split(" ")
            gt_age.append(int(labels[16]))
            gt_em.append(int(labels[17]))
            gt_gen.append(int(labels[15]))
            label=np.array(labels[1:5])
            gt_bb=label.reshape(2,2).astype(float)
            gt_bb = denormalize_landmarks(gt_bb,image_width, image_height)
            gt_bbs.append(gt_bb)





    results = model.predict(source=image, imgsz=640)
    result = results[0].cpu().numpy()
    box=result.boxes.boxes
    bbs=[]
    image = cv2.imread(image)
    d=len(box)
    for i in range(len(box)):
        dd=box[i]
        if len(box)!=0:
            x1, y1, x2, y2,confidence, label = box[i][0],box[i][1],box[i][2],box[i][3],box[i][4],box[i][5]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            width = (box[i][2] - box[i][0])
            height = (box[i][3] - box[i][1]) 
            x_center = (box[i][0] + width/2)
            y_center = (box[i][1] + height/2)

            # Draw the bounding box on the image
            color = (0, 255, 0)  # You can change the color (BGR format)
            thickness = 2  # You can change the thickness
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            text = f'Confidence: {confidence:.2f}'

            if label==0:
                label_t='human'
            elif label==1:
                label_t='animal'
            elif label==2:
                label_t='cartoon'

            text_label = f'Label: '+label_t

            cv2.putText(image, text, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(image, text_label, (0, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            kpt=result.keypoints.data
            kp_x1, kp_y1, kp_x2, kp_y2,kp_x3, kp_y3,kp_x4, kp_y4,kp_x5, kp_y5 = kpt[i][0][0],kpt[i][0][1],kpt[i][1][0],kpt[i][1][1],kpt[i][2][0],kpt[i][2][1],kpt[i][3][0],kpt[i][3][1],kpt[i][4][0],kpt[i][4][1]
            landmarks = [(kp_x1, kp_y1), (kp_x2, kp_y2), (kp_x3, kp_y3), (kp_x4, kp_y4), (kp_x5, kp_y5)]
            color = (0, 255, 0) 
            for landmark in landmarks:
                cv2.circle(image, (int(landmark[0]), int(landmark[1])), 3, color, -1)

            mtl=results[0].mtl
            pred_AGE=mtl[i][3:4]
            pred_age_ev.append(pred_AGE)
            if int(mtl[i][3:4][0])<100:
                age=str(int(mtl[i][3:4][0]))
            else:
                age="unsure"
            
            pred_GEN=mtl[i][0:3]
            
            class_GEN = np.argmax(pred_GEN.cpu())
            pred_gen_ev.append(class_GEN)
            class_labels_GEN = ['female', 'male', 'unsure']
            predicted_class_GEN = class_labels_GEN[class_GEN]

            pred_EM=mtl[i][4:]
            class_EM = np.argmax(pred_EM.cpu())
            pred_em_ev.append(class_EM)
            class_labels_EM = ['angry', 'happy', 'fear', 'sad', 'surprise', 'disgust', 'neutral','unsure']
            predicted_class_EM = class_labels_EM[class_EM]
            bbs.append([[x_center,y_center],[width,height]])


    if show:
        cv2.putText(image, 'Emotion: '+predicted_class_EM, (0, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(image, 'Gender: '+predicted_class_GEN, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(image, 'Age: '+age, (0, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # Display the image with bounding box
        cv2.imshow('Image with Bounding Box', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return pred_age_ev,pred_em_ev,pred_gen_ev,gt_age,gt_em,gt_gen,bbs,gt_bbs


dir='affectnet/test/images/'
metric="emotions"
pred_age=[]
pred_gen=[]
pred_em=[]
em_gt=[]
gen_gt=[]
age_gt=[]


for i in os.listdir(dir):
    pred_age_evs,pred_em_evs,pred_gen_evs,gt_age,gt_em,gt_gen,pred_bbs,gt_bb=test(dir+i,show=False)
    sort_gt,sort_pred,index_gt,index_pred=match(gt_bb,pred_bbs)
    if len(pred_age_evs)==0:continue
    if len(sort_pred)==0:continue
    if len(pred_gen_evs)==0:continue
    if len(pred_em_evs)==0:continue
    for i,j in zip(index_pred,index_gt):
        pred_age_ev=pred_age_evs[i].item()
        pred_age.append(pred_age_ev)
        age_gt.append(gt_age[j])
        
        pred_gen_ev=pred_gen_evs[i].item()
        pred_gen.append(pred_gen_ev)
        
        pred_em_ev=pred_em_evs[i].item()

        pred_em.append(pred_em_ev)
        em_gt.append(gt_em[j])
        gen_gt.append(gt_gen[j])



if metric=="age":

    mae = mean_absolute_error(age_gt, pred_age)
    print("Mean Absolute Error (MAE) for Age Prediction:", mae)
if metric=="emotions":
    column_names=['angry', 'happy', 'fear', 'sad', 'surprise', 'disgust', 'neutral']
    confusion_mat_emotion = confusion_matrix(em_gt, pred_em)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat_emotion, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("confusion_matrix_emotion.png")
    print("Confusion Matrix (Emotion Classification):")
    print(classification_report(em_gt, pred_em))
    num_classes = len(column_names)
    class_accuracy = np.zeros(num_classes)
    for i in range(num_classes):
        total_samples = np.sum(confusion_mat_emotion[i, :])
        correct_samples = confusion_mat_emotion[i, i]
        class_accuracy[i] = correct_samples / total_samples
    for i in range(num_classes):
        print(f"Accuracy for {column_names[i]}: {class_accuracy[i]*100:.2f}%")
    
    
if metric=="gender":
    gender_accuracy = accuracy_score(gen_gt, pred_gen)
    print("Gender Accuracy:", gender_accuracy)



