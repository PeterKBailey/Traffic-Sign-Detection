import 'dart:io';
import 'package:flutter/services.dart';
import 'package:flutter_pytorch/flutter_pytorch.dart';
import 'package:flutter_pytorch/pigeon.dart';
import 'package:flutter/services.dart' show rootBundle;

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
    print("THIS IS THE THING RUNNING HERE *****************************");
    // get prediction
    List<ResultObjectDetection?> objDetect = await _model.getImagePrediction(
      image.readAsBytesSync(),
      minimumScore: 0.1, IOUThershold: 0.3
    );
    Widget newImage = _model.renderBoxesOnImage(image, objDetect);
    return newImage;
  }
}