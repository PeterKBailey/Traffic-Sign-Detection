# Group 99: Traffic-Sign-Detection
Traffic sign recognition and localization.  
- Peter Bailey 101157705  
- Hamza Osman 123123123  
- Vadim Boshnyak 123123123  

## Overview
There are a few key components to the project
1. Our custom AlexNet traffic sign categorization neural network
2. The YOLOv5 training neural network
3. The Flutter mobile app

## Training YOLOv5
1. download the mapillary training and validation datasets from https://www.mapillary.com/dataset/trafficsign
![alt text](image.png)
2. Unzip the files and create the following directory structure:  
```main_directory/  
│  
├── train/  
│   ├── images/  
│       └── [series_of_jpgs]  
│  
├── validate/  
│   ├── images/  
│       └── [series_of_jpgs]  
│  
└── mtsd_v2_fully_annotated/  
    ├── annotations/  
    │   └── [series_of_jsons]  
    └── splits/  
        ├── train.txt  
        ├── test.txt  
        └── val.txt  
```
3. Using the following command to get the data in a structure for YOLO training  
`python /path/to/preprocessing.py initialize --input-location /location/of/data/created/in/step_2 --output-location /where/to/store/processed/data/for/YOLOv5`

4. Now install yolo in `/where/to/store/processed/data/for/YOLOv5/"data"` (note that there is a data directory now)  
```
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

6. Now from within the same directory (`/where/to/store/processed/data/for/YOLOv5/"data"`) run YOLOv5 with  
From within your --output-location run the following command  
`python yolov5/train.py --img 640 --batch 16 --epochs 3 --data dataset.yaml --weights '' --cfg yolov5n.yaml`

Note that the yaml file can be 5n, 5s, 5m and so on.

7. Now generate the torchscript file  
`python yolov5/export.py --weights yolov5/runs/train/expXYZ/weights/best.pt --include torchscript --img 640 --optimize`