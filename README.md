# Group 99: Traffic-Sign-Detection
Traffic sign recognition and localization.  
- Peter Bailey 101157705  
- Hamza Osman 123123123  
- Vadim Boshnyak 123123123  

## Overview
There are a few key components to the project
1. Our custom LeNet and VGGnet traffic sign categorization neural networks
2. The YOLOv5 training neural network
3. The Flutter mobile app

# How To Run

## GTSRB Classification Sample Script

## Android Application
The android app allows for detection and classification in real time while also providing the ability to take snapshots and view them.

Requires either
- A) An Android phone which can install [the app APK](https://drive.google.com/file/d/1MHEOfz43j-FumzRXSGm3AV8wUzSVJSUp/view?usp=sharing)
  1. Download the APK
  2. Install it
  3. Open the application (the prediction models are already included)

- B) An emulated Android phone (Can be obtained through Android Studio go to Device Manager then click "Create device")
  - In this case, follow the build steps decribed in [the section below](#building-android-app)
  - This is not ideal as the purpose is to use the device camera

### How to Use
There are a few pieces to the app
### Detection Screen
* You will see a camera preview, this is the image stream on which the categorization/detection occurs
* The camera button on the bottom center takes a snapshot of the image + label(s) which are viewable in the gallery
* the circular button on the bottom right changes between categorization and identification mode
* Prediction happens automatically, the label will appear on the top for categorization or will display a bounding box for localization

![alt text](Readme-Images/image-5.png) ![alt text](Readme-Images/image-6.png)

### Gallery Screen
* Clicking Gallery on the bottom navigator will show you the snapshots you have taken thus far
* You can tap on one to view it
* This image is mobile, feel free to scale, translate, and rotate it as you like!
* Hit your phone's back button / gesture to return to the gallery

![alt text](Readme-Images/image-2.png) ![alt text](Readme-Images/image-3.png) ![alt text](Readme-Images/image-4.png)


## Building Android App
### Requirements
- Android studio
- An Android phone with developer settings and USB debugging enabled

### Building and Running Project
1. Install android studio: https://developer.android.com/studio/install
2. Install flutter: https://docs.flutter.dev/get-started/editor?tab=androidstudio
3. Open the project
4. Plug developer enabled (with USB-debugging) Android phone into PC
5. Main can be found in the `lib` directory
6. Hit the green arrow from main or the top right
7. Your editor should now look like so:

![alt text](Readme-Images/image-1.png)
