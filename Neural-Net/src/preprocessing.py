import argparse
import json
import numpy as np
import albumentations
import os
from PIL import Image
import cv2

parser = argparse.ArgumentParser(description='Data preprocessing functions')

# Define command line arguments
parser.add_argument('command', type=str, help='The action to perform', choices=["resize-images", "create-label-map", "create-yolotxt"])
parser.add_argument('--output-location', type=str, help='Path to directory where output data will go.')

# mapillary data
parser.add_argument('--annotation-location', type=str, help='Path to the JSON markup files.')
parser.add_argument('--image-location', type=str, help='Path to the image files.')

# resize-images
parser.add_argument('--size', type=int, help='New pixel size of images being squared.')
parser.add_argument('--preserve-aspect', type=bool, help='Whether to preserve the aspect ratio with padding.', default=True)

# convert-to-yolotxt
parser.add_argument('--map-location', type=str, help='Path to JSON file with map between class labels to numbers.')



# Parse the command line arguments
args = parser.parse_args()


def resize_images(image_location:str, annotation_location:str, output_location:str, size: int, preserve_aspect=False):
    '''
    Resizes all images and their bounding box annotation jsons to the specified size, using padding as needed to maintain aspect ratio.
    '''

    print("Resizing images.")

    image_output_location = os.path.join(output_location, "images")
    json_output_location = os.path.join(output_location, "annotations")

    os.mkdir(image_output_location)
    os.mkdir(json_output_location)

    # transformation pipeline (resize) with bboxes (xmin, ymin, xmax, ymax, class)
    if preserve_aspect:
        transform = albumentations.Compose(
            [albumentations.LongestMaxSize(max_size=size, always_apply=True), 
            albumentations.PadIfNeeded(size, size, position=albumentations.augmentations.geometric.transforms.PadIfNeeded.PositionType.CENTER, border_mode=cv2.BORDER_CONSTANT, value=0)],
            bbox_params= albumentations.BboxParams(format='pascal_voc')
        )
    else:
        transform = albumentations.Compose(
            [albumentations.Resize(height=size, width=size, always_apply=True)], 
            bbox_params= albumentations.BboxParams(format='pascal_voc')
        )


    # iterate over json files
    for filename in os.listdir(annotation_location):
        print("PROCESSING THE FILE ", filename)
        # get the old data and set new info
        og = open(os.path.join(annotation_location, filename))
        annotation = json.load(og)

        # read the boxes in
        boxes = []
        for box in annotation["objects"]:
            boxes.append([box["bbox"]["xmin"], box["bbox"]["ymin"], box["bbox"]["xmax"], box["bbox"]["ymax"], 0])

        # get the corresponding image and boxes in numpy
        related_img = Image.open(os.path.join(image_location, filename[0:-5] + ".jpg"))
        img_arr = np.asarray(related_img)
        bboxes = np.array(boxes)

        # perform resize and get new boxes
        transformed = transform(image=img_arr, bboxes=bboxes)
        
        # update the image size in JSON
        annotation["width"] = size
        annotation["height"] = size

        # update the boxes in JSON
        newBoxData = transformed["bboxes"]
        for i in range(0, len(annotation["objects"])):
            box = annotation["objects"][i]["bbox"]
            box["xmin"] = newBoxData[i][0]
            box["ymin"] = newBoxData[i][1]
            box["xmax"] = newBoxData[i][2]
            box["ymax"] = newBoxData[i][3]

        # save the new JSON data 
        newJSON = open(os.path.join(json_output_location, filename), "w")
        json.dump(annotation, newJSON)

        # save the new image data
        Image.fromarray(transformed["image"]).save(os.path.join(image_output_location, filename[0:-5] + ".jpg"))
        
        # cleanup
        og.close()
        newJSON.close()


def create_label_map(annotation_location: str, output_location:str):
    '''
    Creates a map from label to integer by checking every type of sign in the annotations.
    '''
    print("Creating label map.")
    labels = {}
    classId = 0

    # iterate over json files
    for filename in os.listdir(annotation_location):
        print("INVESTIGATING FILE ", filename)
        # read in the annotation file
        og = open(os.path.join(annotation_location, filename))
        annotation = json.load(og)

        for sign in annotation["objects"]:
            label = sign["label"]
            if(label not in labels):
                labels[label] = classId
                classId += 1
    
    labelMapFile = open(os.path.join(output_location, "label_map.json"), "w")
    json.dump(labels, labelMapFile)
    labelMapFile.close()
        

def create_yolotxt(image_location: str, annotation_location:str, output_location:str, label_map_location:str):
    '''
    Creates a txt file for each image with the bounding boxes and categories.
    '''
    print("Creating yolotxt representation.")
    # so you need to get the .txt file for each image, each line is a box and category from the json for that image
    # i was gona say you need to rename them but actually i think theyre already unique and matching anyway, it should just work
    # so maybe if possible this can do all the work in terms of setting up the directories too?
        # IG this function could be run twice, once for:
        # yolov5_ws/data/images/training/
        # yolov5_ws/data/labels/training/

        # and once more for:
        # yolov5_ws/data/images/validation/
        # yolov5_ws/data/labels/validation/

    # iterate over json files
    label_categories = json.load(label_map_location)

    for filename in os.listdir(annotation_location):
        print("YOLOing FILE ", filename)
        # read in the annotation file
        og = open(os.path.join(annotation_location, filename))
        annotation = json.load(og)

        yolo_labels = []
        for sign in annotation["objects"]:
            category = label_categories[sign["label"]]
            yolo_labels.append(f'{sign["bbox"]["xmin"]} {sign["bbox"]["ymin"]} {sign["bbox"]["xmax"]} {sign["bbox"]["ymax"]} {category}=')
        
        
        yolotxt = open(os.path.join(output_location, filename+".txt"), "w")
        yolotxt.write(yolo_labels.join("\n"))
        yolotxt.close()
       


if args.command == "resize-images":
    resize_images(args.image_location, args.annotation_location, args.output_location, args.size, args.preserve_aspect)
elif args.command == "create-label-map":
    create_label_map(args.annotation_location, args.output_location)
elif args.command == "create-yolotxt":
    create_yolotxt(args.image_location, args.annotation_location, args.output_location)