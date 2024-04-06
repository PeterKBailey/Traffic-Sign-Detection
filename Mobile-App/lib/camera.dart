import 'dart:async';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_pytorch/flutter_pytorch.dart';
import 'package:flutter_pytorch/pigeon.dart';
import 'package:screenshot/screenshot.dart';
import 'package:gal/gal.dart';

class Camera extends StatefulWidget{
  final Function(Uint8List) _imageHandler;

  const Camera(Function(Uint8List) imageHandler, {super.key}): _imageHandler = imageHandler;

  @override
  State<StatefulWidget> createState() {
    return _CameraState();
  }
}

class _CameraState extends State<Camera> {
  // for the camera
  CameraController? _cameraController;
  // for capturing images
  ScreenshotController screenshotController = ScreenshotController();

  // detection
  CameraImage? _frame;
  Timer? _timer;
  bool _objectDetectionMode = false;

  // model
  ModelObjectDetection? _model;
  ClassificationModel? _classificationModel;
  final int _numClasses = 80;
  // final int _numClasses = 401;

  // detection results
  List<ResultObjectDetection?>? _detectionResult;
  String? _classificationResult;

  _CameraState(){
    _start();
  }

  Future<void> _start() async {
    // first: get access to the camera
    List<CameraDescription> cameras = await availableCameras();
    final cameraController = CameraController(
        cameras[0],
        ResolutionPreset.high,
        enableAudio: false
    );

    await cameraController.initialize();
    _cameraController = cameraController;

    // second: load in the models
    _model = await FlutterPytorch.loadObjectDetectionModel(
        "assets/processing/yolov5s.torchscript", _numClasses, 640, 640,
        labelPath: "assets/processing/labels_objectDetection_Coco.txt");

    _classificationModel = await FlutterPytorch.loadClassificationModel(
        "assets/processing/model.pt", 32, 32,
        labelPath: "assets/processing/labels_gtsrb.txt");

    // third: access the camera frames, store the latest one
    cameraController.startImageStream((CameraImage image) async {
      _frame = image;
    }).then((_){
      // fourth: run detection once the camera is running
      runDetection();
    });
  }

  void runDetection() async {
    // The timer speed is the minimum amount of time between detections
    // the fact is that processing can take longer than this amount of time
    _timer = Timer.periodic(const Duration(milliseconds: 24), (Timer t) async {
      var frame = _frame;
      if(frame == null){
        return;
      }
      // important: detection is costly, if we don't pause the timer then
      // the timer callbacks can outpace the speed of detection causing delays
      // (we don't want the timer making more callbacks while the prev detection is still going)
      t.cancel();

      // print("now predicting...");
      if(_objectDetectionMode){
        List<ResultObjectDetection?>? detectionResult = await _model
            ?.getImagePredictionFromBytesList(
            frame.planes.map((e) => e.bytes).toList(),
            frame.width,
            frame.height,
            minimumScore: 0.3,
            IOUThershold: 0.3
        );

        setState(() {
          _detectionResult = detectionResult;
        });
      }
      else{
        String? classificationResult = await _classificationModel
            ?.getImagePredictionFromBytesList(
            frame.planes.map((e) => e.bytes).toList(),
            frame.width,
            frame.height
        );

        setState(() {
          _classificationResult = classificationResult;
        });
      }

      // restart the timer
      runDetection();
    });
  }

  void takePicture() {
    screenshotController.capture(
        delay: const Duration(milliseconds: 0)
    ).then((capturedImage){
      if(capturedImage == null){
        return;
      }

      widget._imageHandler(capturedImage);
      // Gal.putImageBytes(capturedImage);
    });
  }

  @override
  void dispose() {
    // clean up camera and stop recognition timer
    Future.delayed(Duration.zero, () async {
      await _cameraController?.stopImageStream();
      await _cameraController?.dispose();
    }).then((value){
      _timer?.cancel();
    });
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    List<ResultObjectDetection?>? detectionResult = _detectionResult;

    return LayoutBuilder(builder: (context, constraints) {
      final CameraController? cameraController = _cameraController;

      // while waiting for the camera return loading circle
      if (cameraController == null || !cameraController.value.isInitialized) {
        return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              // display a loading circle
              children: [
                Container(
                    padding: const EdgeInsets.only(bottom: 25),
                    child: const Text("Loading")
                ),
                const CircularProgressIndicator()
              ],
            )
        );
      }

      List<Widget> screenshotChildren = [];

      if(_objectDetectionMode){
        // Add the full size camera preview
        screenshotChildren.add(
            LayoutBuilder(
                builder: (context, constraints) {
                  return SizedBox(
                      width: constraints.maxWidth,
                      height: constraints.maxHeight,
                      child: CameraPreview(cameraController)
                  );
                }
            )
        );

        // need boxes to fit on the screen in the appropriate size / place
        double factorX = constraints.maxWidth;
        double factorY = constraints.maxHeight;
        if(detectionResult != null) {
          // add the bounding boxes
          screenshotChildren.addAll(
            // take all the detections and make a box widget for each
            detectionResult.map((re) {
              if (re == null) {
                return Container();
              }
              // https://math.stackexchange.com/a/914843
              // map category to colour
              Color boxColour = Colors.primaries[
              (0 +
                  (Colors.primaries.length - 1 - 0) / (_numClasses - 1 - 0)
                      * (re.classIndex)
              ).round()
              ];
              // the detection
              return Positioned(
                left: re.rect.left * factorX,
                // -20 accounts for 20 pixels added to show the class name
                top: re.rect.top * factorY - 20,

                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  mainAxisAlignment: MainAxisAlignment.start,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // the label
                    Container(
                      height: 20,
                      color: boxColour,
                      child: Text(
                          "${re.className ?? re.classIndex.toString()} ${(re
                              .score * 100).toStringAsFixed(2)}%"
                      ),
                    ),
                    // the box itself
                    Container(
                      width: re.rect.width.toDouble() * factorX,
                      height: re.rect.height.toDouble() * factorY,
                      decoration: BoxDecoration(
                          color: boxColour.withOpacity(0.3),
                          border: Border.all(color: boxColour, width: 3),
                          borderRadius: const BorderRadius.all(
                              Radius.circular(10))
                      ),
                      child: Container(),
                    ),
                  ],
                ),
              );
            }
            )
          );
        }
      }
      // otherwise were doing categorization, show a camera box
      else{
        double previewSize = MediaQuery.of(context).size.height / 3;
        // add the square box live video
        screenshotChildren.add(
          Center(child: SizedBox(
            width: previewSize,
            height: previewSize,
            child: ClipRect(
              child: OverflowBox(
                alignment: Alignment.center,
                child: FittedBox(
                  fit: BoxFit.fitWidth,
                  child: SizedBox(
                    width: previewSize,
                    height: previewSize*cameraController.value.aspectRatio,
                    child: CameraPreview(cameraController),
                  ),
                ),
              ),
            ),
          ))
        );

        // add the classification label text
        screenshotChildren.add(
          Align(
              alignment: Alignment.topCenter,
              child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Container(
                      decoration: BoxDecoration(
                          color: Theme.of(context).colorScheme.background.withOpacity(0.3),
                          border: Border.all(color: Theme.of(context).colorScheme.background, width: 3),
                          borderRadius: const BorderRadius.all(Radius.circular(10))
                      ),
                      child: Text(_classificationResult ?? "",
                          textAlign: TextAlign.center,
                          style: TextStyle(color: Theme.of(context).colorScheme.primary, fontSize:32)
                      )
                  )
              )
          )
        );
      }

      // finally put it all together
      return Stack(
        children: [
          // this is what will appear when photo taken
          Screenshot(
            controller: screenshotController,
            child: Stack(
              children: screenshotChildren
            )
          ),

          // the capture photo button
          Align(
            alignment: Alignment.bottomCenter,

            child: Padding(
              padding: const EdgeInsets.all(20),
              child: FloatingActionButton(

                onPressed: takePicture,
                child: const Icon(Icons.camera_alt)
              )
            )
          ),

          // the mode switch button
          Align(
              alignment: Alignment.bottomRight,

              child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: FloatingActionButton(
                    shape: const CircleBorder(side:BorderSide()),
                    onPressed: (){
                      _objectDetectionMode = !_objectDetectionMode;
                    },
                    child: const Icon(Icons.track_changes_outlined)
                  )
              )
          )
        ]
      );
    });
  }
}