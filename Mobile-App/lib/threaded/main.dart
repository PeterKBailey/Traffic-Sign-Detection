import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_pytorch/flutter_pytorch.dart';
import 'package:flutter_pytorch/pigeon.dart';
import 'package:screenshot/screenshot.dart';
import 'package:gal/gal.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<StatefulWidget> createState() {
    return _MyAppState();
  }
}

class _MyAppState extends State<MyApp> with SingleTickerProviderStateMixin{
  ScreenshotController screenshotController = ScreenshotController();
  final _camera = const Camera();

  _MyAppState();

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        title: 'BBB Traffic Sign Detector',
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
          useMaterial3: true,
        ),
        home: Scaffold(
          appBar: AppBar(
            backgroundColor: Theme.of(context).colorScheme.inversePrimary,
            title: const Text("Traffic Sign Detection!"),
          ),
          body: Screenshot(controller: screenshotController, child: _camera),
          floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
          floatingActionButton: TakePictureButton(screenshotController: screenshotController,)
        )
    );
  }
}

class TakePictureButton extends StatefulWidget{
  final ScreenshotController screenshotController;
  const TakePictureButton({super.key, required this.screenshotController});

  @override
  State<StatefulWidget> createState() {
    return _TakePictureButtonState();
  }

}

class _TakePictureButtonState extends State<TakePictureButton>{
  @override
  Widget build(BuildContext context) {
    return FloatingActionButton(
      child: const Icon(Icons.camera_alt),
      onPressed: (){
        ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text("Picture taken!"))
        );
        widget.screenshotController.capture(
            delay: const Duration(milliseconds: 0)
        ).then((capturedImage) async {
          print("Captured in then");
          if(capturedImage == null){
            print("The image was not captured!!");
            return;
          }
          ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text("Picture saved!"))
          );
          await Gal.putImageBytes(capturedImage);
        });
      },
    );
  }

}


class Camera extends StatefulWidget{
  const Camera({
    super.key,
  });

  @override
  State<StatefulWidget> createState() {
    return _CameraState();
  }
}

class _CameraState extends State<Camera> {
  CameraController? _cameraController;
  CameraImage? _frame;
  final int _numClasses = 80;

  List<ResultObjectDetection?>? _detectionResult;
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
        // imageFormatGroup: ImageFormatGroup.nv21
    );

    await cameraController.initialize();
    _cameraController = cameraController;

    // second: load in the model
    var model = await FlutterPytorch.loadObjectDetectionModel(
        "assets/processing/yolov5s.torchscript", _numClasses, 640, 640,
        labelPath: "assets/processing/labels_objectDetection_Coco.txt");

    // third: access the camera frames, store the latest one
    cameraController.startImageStream((CameraImage image) async {
      _frame = image;
    }).then((_){
      // fourth: run detection
      runDetection(model);
    });

  }

  void runDetection(ModelObjectDetection model) async {
    // every x ms run detection
    // TODO: disposal of cameraController and the timer?
    var timer = Timer.periodic(const Duration(milliseconds: 1), (Timer t) async {
      var frame = _frame;
      if(frame == null){
        return;
      }
      // important: detection is costly, if we don't pause the timer then
      // the timer callbacks can outpace the speed of detection causing delays
      t.cancel();

      // print("now predicting...");
      List<ResultObjectDetection?> result = await model
          .getImagePredictionFromBytesList(
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
      runDetection(model);
    });
  }


  @override
  // Code based on flutter pytorch
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
      double factorX = constraints.maxWidth;
      double factorY = constraints.maxHeight;
      return Stack(
        children: [
          CameraPreview(cameraController),
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
          })
        ],
      );
    });
  }
}
