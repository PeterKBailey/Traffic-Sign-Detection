import 'dart:async';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
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
  CameraController? _cameraController;
  ScreenshotController screenshotController = ScreenshotController();

  Timer? _timer;
  ModelObjectDetection? _model;
  CameraImage? _frame;
  final int _numClasses = 80;

  List<ResultObjectDetection?>? _detectionResult;
  _CameraState(){
    print("WHY IS THIS HAPPENING AGAIN");
    _start();
  }

  Future<void> _start() async {
    // first: get access to the camera
    List<CameraDescription> cameras = await availableCameras();
    final cameraController = CameraController(
        cameras[0],
        ResolutionPreset.high,
        enableAudio: false
      // imageFormatGroup: ImageFormatGroup.nv21
    );

    await cameraController.initialize();
    _cameraController = cameraController;

    // second: load in the model
    _model = await FlutterPytorch.loadObjectDetectionModel(
        "assets/processing/yolov5s.torchscript", _numClasses, 640, 640,
        labelPath: "assets/processing/labels_objectDetection_Coco.txt");

    // third: access the camera frames, store the latest one
    cameraController.startImageStream((CameraImage image) async {
      _frame = image;
    }).then((_){
      // fourth: run detection
      runDetection();
    });

  }

  void runDetection() async {
    // every x ms run detection
    _timer = Timer.periodic(const Duration(milliseconds: 1), (Timer t) async {
      var frame = _frame;
      if(frame == null){
        return;
      }
      // important: detection is costly, if we don't pause the timer then
      // the timer callbacks can outpace the speed of detection causing delays
      t.cancel();

      // print("now predicting...");
      List<ResultObjectDetection?>? result = await _model
          ?.getImagePredictionFromBytesList(
          frame.planes.map((e) => e.bytes).toList(),
          frame.width,
          frame.height,
          minimumScore: 0.3,
          IOUThershold: 0.3);

      setState(() {
        // print("setting state");
        _detectionResult = result;
      });

      // restart the timer
      runDetection();
    });
  }

  void takePicture() {
    // ScaffoldMessenger.of(context).showSnackBar(
    //     const SnackBar(content: Text("Picture taken!"))
    // );

    screenshotController.capture(
        delay: const Duration(milliseconds: 0)
    ).then((capturedImage){
      if(capturedImage == null){
        return;
      }
      // if (context.mounted){
      //   ScaffoldMessenger.of(context).showSnackBar(
      //       const SnackBar(content: Text("Picture saved!"))
      //   );
      // }
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

    // want to draw the
    return LayoutBuilder(builder: (context, constraints) {
      final CameraController? cameraController = _cameraController;

      // while waiting for the camera and model
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
      // boxes fit on the screen in the appropriate size / place
      double factorX = constraints.maxWidth;
      double factorY = constraints.maxHeight;
      return Stack(
        children: [
          Screenshot(
            controller: screenshotController,
            child: Stack(
              children: [
                // The live video
                LayoutBuilder(
                    builder: (context, constraints) {
                      return SizedBox(
                          width: constraints.maxWidth,
                          height: constraints.maxHeight,
                          child: CameraPreview(cameraController)
                      );
                    }
                ),

                // the bounding boxes
                // Code based on flutter pytorch
                // https://pub.dev/packages/flutter_pytorch
                ...?detectionResult?.map((re) {
                  if (re == null) {
                    return Container();
                  }
                  // https://math.stackexchange.com/a/914843
                  // map category to colour
                  Color boxColour = Colors.primaries[
                  (0 +
                      (Colors.primaries.length-1 - 0)/(_numClasses-1 - 0)
                          *(re.classIndex)
                  ).round()
                  ];

                  return Positioned(
                    left: re.rect.left * factorX,
                    // -20 accounts for 20 added to show the class name
                    top: re.rect.top * factorY - 20,

                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      mainAxisAlignment: MainAxisAlignment.start,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Container(
                          height: 20,
                          color: boxColour,
                          child: Text(
                              "${re.className ?? re.classIndex.toString()} ${(re.score * 100).toStringAsFixed(2)}%"
                          ),
                        ),
                        Container(
                          width: re.rect.width.toDouble() * factorX,
                          height: re.rect.height.toDouble() * factorY,
                          decoration: BoxDecoration(
                              color: boxColour.withOpacity(0.3),
                              border: Border.all(color: boxColour, width: 3),
                              borderRadius: const BorderRadius.all(Radius.circular(10))
                          ),
                          child: Container(),
                        ),
                      ],
                    ),
                  );
                }),
              ],
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
          )
        ]
      );
    });
  }
}