import 'dart:io';
import 'package:flutter_pytorch/flutter_pytorch.dart';
import 'package:flutter_pytorch/pigeon.dart';

// Widget?
import 'package:flutter/material.dart';



class ImageProcessor {
  late ModelObjectDetection _model;
  // private ctor
  ImageProcessor._create();

  /// public factory
  static Future<ImageProcessor> create(String pathToModel) async {
    print("There is creation...");
    var processor = ImageProcessor._create();
    processor._model = await FlutterPytorch.loadObjectDetectionModel(
        pathToModel, 80, 640, 640,
        labelPath: "assets/processing/labels_objectDetection_Coco.txt");
    // Return the fully initialized object
    return processor;
  }

  // run an image model
  Future<Widget> run(File image) async {
    // return Image(image: FileImage(image));

    print("THIS IS THE THING RUNNING HERE *****************************");
    Stopwatch stopwatch = Stopwatch()..start();

    // get prediction
    List<ResultObjectDetection?> objDetect = await _model.getImagePrediction(
      image.readAsBytesSync(),
      minimumScore: 0.1, IOUThershold: 0.3
    );
    Widget newImage = _model.renderBoxesOnImage(image, objDetect);
    print('processing executed in ${stopwatch.elapsed.inMilliseconds} miliseconds');

    return newImage;
  }
}