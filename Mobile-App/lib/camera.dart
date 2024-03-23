import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'dart:io';

class Camera extends StatefulWidget{
  final Function(File) callback;

  const Camera({super.key, required this.callback});

  @override
  State<Camera> createState() {
    return _CameraState();
  }
}

class _CameraState extends State<Camera> {
  late CameraController controller;

  @override
  void initState() {
    super.initState();
    initCamera();
  }

  Future<void> initCamera() async {
    List<CameraDescription> cameras = await availableCameras();

    controller = CameraController(cameras[0], ResolutionPreset.high);
    controller.initialize().then((_) {
      if (!mounted) {
        return;
      }
      // rebuild
      setState(() {});
    }).catchError((Object e) {
      if (e is CameraException) {
        switch (e.code) {
          case 'CameraAccessDenied':
          // Handle access errors here.
            break;
          default:
          // Handle other errors here.
            break;
        }
      }
    });
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!controller.value.isInitialized) {
      return Container();
    }
    return
      Column(
        children: [
          Container(
            width: MediaQuery.of(context).size.height / 3,
            height: MediaQuery.of(context).size.height / 3,
            child: ClipRect(
              child: OverflowBox(
                alignment: Alignment.center,
                child: FittedBox(
                  fit: BoxFit.fitWidth,
                  child: Container(
                    width: MediaQuery.of(context).size.height / 3,
                    height: MediaQuery.of(context).size.height / 3,
                    child: CameraPreview(controller), // this is my CameraPreview
                  ),
                ),
              ),
            ),
          ),
          // Expanded(child: CameraPreview(controller)),
          ElevatedButton(
            // Provide an onPressed callback.
            onPressed: () async {
              // Take the Picture in a try / catch block. If anything goes wrong,
              // catch the error.
              try {
                // Attempt to take a picture and then get the location
                // where the image file is saved.
                final image = await controller.takePicture();

                widget.callback(File(image.path));
              } catch (e) {
                // If an error occurs, log the error to the console.
                print(e);
              }
            },
            child: const Icon(Icons.camera_alt),
          )
      ]
    );
  }
}