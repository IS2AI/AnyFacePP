# AnyFace++: Deep Multi-Task Multi-Domain Learning for Efficient Face AI ([Paper](https://www.mdpi.com/1424-8220/24/18/5993))
![Anyfacepp](https://github.com/IS2AI/AnyFacePP/blob/main/predictions.png)

## Installation requirements

Please clone our repository, even if you already have the original Yolov8 files. Our model will not function properly due to significant modifications in the code.
Clone the repository and install all necessary packages. Please ensure that Python>=3.8 with PyTorch>=1.8.
```
git clone https://github.com/IS2AI/AnyFacePP.git
cd AnyFacePP
pip install ultralytics
```
The following datasets were used to train, validate, and test the models.

| Dataset | Link    |
| :---:   | :---: | 
| Facial Expression Recognition 2013 | https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data  |
| AffectNet | http://mohammadmahoor.com/affectnet/  |
| IMDB | https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/  |
| UTKFace | https://susanqq.github.io/UTKFace/  |
| Adience | https://talhassner.github.io/home/projects/Adience/Adience-data.html |
| MegaAge | http://mmlab.ie.cuhk.edu.hk/projects/MegaAge/  |
| MegaAge Asian | http://mmlab.ie.cuhk.edu.hk/projects/MegaAge/  | 
| AFAD Dataset | https://afad-dataset.github.io/  |
| AgeDB | https://complexity.cecs.ucf.edu/agedb/ |
| FairFace | https://github.com/joojs/fairface |
| Uniform Age and Gender Dataset (UAGD) | https://github.com/clarkong/UAGD  |
| FG-NET | https://yanweifu.github.io/FG_NET_data/  | 
| RAF-DB (Real-world Affective Faces) |  http://www.whdeng.cn/raf/model1.html | 
| Wider Face | http://shuoyang1213.me/WIDERFACE/  |
| AnimalWeb | 	https://fdmaproject.wordpress.com/author/fdmaproject/  |
| iCartoonFace | https://github.com/luxiangju-PersonAI/iCartoonFace#dataset |
| TFW | https://github.com/IS2AI/TFW#downloading-the-dataset  |

## Preprocessing Step

Use notebooks in the main directory to pre-process the corresponding datasets.

The preprocessed datasets are saved in `dataset/` directory. For each dataset, images are stored in `dataset/<dataset_name>/images/` and the corresponding labels are stored in `dataset/dataset_name/labels/` and in `dataset/<dataset_name>/labels_eval/`. Labels are saved in `.txt` files, where each `.txt` file has the same filename as corresponding image.

Annotations in `dataset/<dataset_name>/labels/` follow the format used for training YOLOv8Face models:

* `face_type x_center y_center width height x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 gender age emotion`
* `face_type`' represents the type of face: 0 - human, 1 - animal, 2 - cartoon.
*  `x1,y1,...,x5,y5` correspond to the coordinates of the left eye, right eye, nose top, left mouth corner, and right mouth corner.
* `gender` denotes the gender of the person: 1 - male, 0 - female, 2 - unsure.
* `age` indicates the age of the person.
* `emotion` specifies one of the 7 basic emotions (0 - angry, 1 - happy, 2 - fear, 3 - sad, 4 - surprise, 5 - disgust, 6 - neutral, -2 - unsure).
All coordinates are normalized to values between 0 and 1. If a face lacks any of the labels, -1 is used in place of the missing values.

## Training Step
Run the following command to train the model on previously specified datasets. The paths to these datasets are defined in the dataset.yaml file. The model type is selected by the line in the code "yolov8m-pose.yaml". The weights are randomly initialized --weights ''. However, pre-trained weights can be also used by providing an appropriate path.
   ```
   python3 train.py
   ```

## Inference

Download the most accurate model, YOLOv8, from [Google Drive](https://drive.google.com/drive/folders/1xOpe26HvIjX2VCLNQmMvecmqHVYniPNM?usp=sharing) and save it. 


   ```
   python3 inference.py
   ```
You can specify parameters in code:

   path_to_model= path to the model
   path_to_image= path to the image location #if you want use image from the Internet, replace path with None
   image_url=None #if you want use image from the Internet, replace None with URL
   threshold_bboxes=0.3 

## Inference in real-time

Please use this code to run a model in real-time detection:
   ```
   python3 inference_real.py
   ```
You can specify parameters in code:
path_to_model= path to the model

You can specify you and confidence in the following line:
model.predict(source="0",show=True,iou=0.2,conf=0.6, device="cuda") 

## In case of using our work in your research, please cite this paper
```
@Article{s24185993,
AUTHOR = {Rakhimzhanova, Tomiris and Kuzdeuov, Askat and Varol, Huseyin Atakan},
TITLE = {AnyFace++: Deep Multi-Task, Multi-Domain Learning for Efficient Face AI},
JOURNAL = {Sensors},
VOLUME = {24},
YEAR = {2024},
NUMBER = {18},
ARTICLE-NUMBER = {5993},
URL = {https://www.mdpi.com/1424-8220/24/18/5993},
ISSN = {1424-8220},
DOI = {10.3390/s24185993}
}

```

## References

* [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

* [https://github.com/deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)

* [https://github.com/derronqi/yolov7-face](https://github.com/derronqi/yolov7-face)
