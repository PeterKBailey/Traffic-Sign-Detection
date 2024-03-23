import 'dart:io';
import 'package:flutter/services.dart';
import 'package:flutter_pytorch/flutter_pytorch.dart';
import 'package:flutter_pytorch/pigeon.dart';



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
  Future<List<dynamic>?> run(File image) async {
    // get prediction
    List<ResultObjectDetection?> objDetect = await _model.getImagePrediction(await image.readAsBytes(),
        minimumScore: 0.1, IOUThershold: 0.3);

    print("Forward pass...");
    return objDetect;
  }
}