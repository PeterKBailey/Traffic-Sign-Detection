import argparse
import json
import numpy as np
import albumentations
import os
from PIL import Image
import cv2
import shutil
import yaml

parser = argparse.ArgumentParser(description='Data preprocessing functions')

# Define command line arguments
parser.add_argument('command', type=str, help='The action to perform', choices=["resize-images", "create-label-map", "create-yolotxt", "initialize"])
parser.add_argument('--output-location', type=str, help='Path to directory where output data will go.')
parser.add_argument('--input-location', type=str, help='Path to directory where input data is found.')

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


def resize_images(image_location:str, annotation_location:str, image_output_location:str, json_output_location: str,  size: int, preserve_aspect=False):
    '''
    Resizes all images and their bounding box annotation jsons to the specified size, using padding as needed to maintain aspect ratio.
    '''

    print("Resizing images.")

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
        

def create_yolotxt(annotation_location:str, output_location:str, label_map_location:str,):
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
    mapfile = open(label_map_location)
    label_categories = json.load(mapfile)
    mapfile.close()

    for filename in os.listdir(annotation_location):
        # read in the annotation file
        og = open(os.path.join(annotation_location, filename))
        annotation = json.load(og)
        og.close()

        yolo_labels = []
        image_width = annotation["width"]
        image_height = annotation["height"] #3024
        for sign in annotation["objects"]:
            category = label_categories[sign["label"]]
            # use the box dimensions to compute yolov5 info
            width = sign["bbox"]["xmax"] - sign["bbox"]["xmin"]
            height = sign["bbox"]["ymax"] - sign["bbox"]["ymin"]
            center_x = sign["bbox"]["xmin"] + width/2
            center_y = sign["bbox"]["ymin"] + height/2
            # now regularize
            yolo_labels.append(f'{category} {center_x/image_width} {center_y/image_height} {width/image_width} {height/image_height}')
        
        
        yolotxt = open(os.path.join(output_location, filename[0:-5]+".txt"), "w")
        yolotxt.write("\n".join(yolo_labels))
        yolotxt.close()
       

def initialize(data_directory: str, output_directory: str, resize = "NONE"):
    '''
    Get's everything in order to start running YOLO
    '''
    if(resize not in ["NONE", "STRETCH", "LETTERBOX"]):
        print("THIS IS NOT A VALID INPUT FOR RESIZE! Program exiting.")
        return

    # 1. check that the file structures are as needed
        # so basically start with a directory containing 
        # -- train
        # ---- images
        # ------ a series of .jpgs
        # -- validate
        # ---- images
        # ------ a series of .jpgs
        # -- test
        # ---- images
        # ------ a series of .jpgs
        # -- mtsd_v2_fully_annotated
        # ---- annotations
        # ------ a series of .jsons
        # ---- splits
        # ------- train.txt, test.txt, val.txt
    
    # this just checks that the dirs exist and there are files in them, no check that the data is right
    dirs = [
        os.path.join(data_directory, "train/images"), 
        os.path.join(data_directory, "validate/images"), 
        # os.path.join(data_directory, "test/images"),
        os.path.join(data_directory, "mtsd_v2_fully_annotated/annotations"),
        os.path.join(data_directory, "mtsd_v2_fully_annotated/splits")
    ]
    for path in dirs:
        print("looking at", path)
        if not os.path.isdir(path) or len(os.listdir(path)) == 0:
            print("THE DATA DIRECTORY STRUCTURE IS NOT AS EXPECTED! Program exiting.")
            return

    # 2. make directories for the labels and put them where they belong

    splits_path = os.path.join(data_directory, "mtsd_v2_fully_annotated/splits")
    annotations_path = os.path.join(data_directory, "mtsd_v2_fully_annotated/annotations")

    # copy the training labels to where needed
    train_label_path = os.path.join(data_directory, "train/labels")
    os.mkdir(train_label_path)

    print("Copying training labels to temp directory.")
    for filename in os.listdir(os.path.join(data_directory, "train/images")):
        related_json_name = filename[0:-4]+".json"
        shutil.copy(os.path.join(annotations_path, related_json_name), os.path.join(train_label_path, related_json_name))


    # copy the validation labels to where needed
    val_label_path = os.path.join(data_directory, "validate/labels")
    os.mkdir(val_label_path)
    
    print("Copying validation labels to temp directory.")
    for filename in os.listdir(os.path.join(data_directory, "validate/images")):
        related_json_name = filename[0:-4]+".json"
        shutil.copy(os.path.join(annotations_path, related_json_name), os.path.join(val_label_path, related_json_name))

# THERE APPEAR TO BE NO TESTING LABELS?
    # copy the testing labels where needed

    # 3. create the label map
        # mapping between the label strings and a category value since its not included for some reason
    create_label_map(os.path.join(data_directory, "mtsd_v2_fully_annotated/annotations"), output_directory)

    # 4. output images and annotations in structure as
        # /data/images/training/       (images)
        # /data/labels/training/       (txt annotations)
        # /data/images/validation/     (images)
        # /data/labels/validation/     (txt annotations)
        # /data/images/testing/        (images)
        # /data/labels/testing/        (txt annotations)

    out_path = os.path.join(output_directory, "data/")

    # transform images and annotations and store in the output dirs
        # put them in the right places...
    if resize != "NONE":
        print("Resizing and moving images to processing directory -", resize)
        resize_images(
            os.path.join(data_directory, "train/images"), 
            train_label_path, 
            os.path.join(out_path, "images/training"), 
            os.path.join(out_path, "labels/training"),
            640,
            preserve_aspect=(resize == "LETTERBOX")
        )
        resize_images(
            os.path.join(data_directory, "validate/images"), 
            val_label_path, 
            os.path.join(out_path, "images/validation"), 
            os.path.join(out_path, "labels/validation"),
            640,
            preserve_aspect=(resize == "LETTERBOX")
        )
        # resize_images(
        #     os.path.join(data_directory, "test/images"), 
        #     test_label_path, 
        #     os.path.join(out_path, "images/testing"), 
        #     os.path.join(out_path, "labels/testing"),
        #     640,
        #     preserve_aspect=(resize == "LETTERBOX")
        # )
    else:
        print("No resize occuring, only moving files to processing directory.")

        shutil.copytree(os.path.join(data_directory, "train/images"), os.path.join(out_path, "images/training"))
        shutil.copytree(train_label_path, os.path.join(out_path, "labels/training"))
        # os.makedirs(os.path.join(out_path, "labels/training"))

        shutil.copytree(os.path.join(data_directory, "validate/images"), os.path.join(out_path, "images/validation"))
        shutil.copytree(val_label_path, os.path.join(out_path, "labels/validation"))
        # os.makedirs(os.path.join(out_path, "labels/validation"))

    # 5. Convert jsons to txt

    label_paths = [
        os.path.join(out_path, "labels/training"),
        os.path.join(out_path, "labels/validation")
    ]

    print("Converting json to txt files")
    label_map_location = os.path.join(output_directory, "label_map.json")
    for path in label_paths:
        # convert to yolotxt, input location -> output location
        create_yolotxt(path, path, label_map_location)
        # clean up the jsons
        for label_file in os.listdir(path):
            if label_file.endswith(".json"):
                os.remove(os.path.join(path, label_file))

    # 6. create the config dataset.yaml
    mapfile = open(label_map_location)
    label_map = json.load(mapfile)
    mapfile.close()

    print("creating dataset.yaml")
    num_categories = len(label_map)
    categories = [None]*num_categories
    for key in label_map:
        categories[label_map[key]] = key
    
    config = {
        "train": os.path.abspath(os.path.join(out_path, "images/training")), 
        "val": os.path.abspath(os.path.join(out_path, "images/validation")),
        # test: os.path.join(out_path, "images/testing"), 
        # number of classes
        "nc": num_categories,
        # class names
        "names": categories
    }
    f = open(os.path.join(out_path, 'dataset.yaml'), 'w+')
    yaml.dump(config, f)
    f.close()

    # 7. now you can run yolo?? right??


if args.command == "resize-images":
    resize_images(args.image_location, args.annotation_location, os.path.join(args.output_location, "images"), os.path.join(args.output_location, "annotations"), args.size, args.preserve_aspect)
elif args.command == "create-label-map":
    create_label_map(args.annotation_location, args.output_location)
elif args.command == "create-yolotxt":
    create_yolotxt(args.annotation_location, args.output_location, args.map_location)
elif args.command == "initialize":
    initialize(args.input_location, args.output_location)