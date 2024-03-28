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
  CameraController? controller;
  bool isAccessDenied = false;

  @override
  void initState() {
    super.initState();
    initCamera();
  }

  Future<void> initCamera() async {
    List<CameraDescription> cameras = await availableCameras();

    final cameraController = CameraController(cameras[0], ResolutionPreset.max);
    // these now reference same object
    controller = cameraController;

    // update ui when controller changes
    cameraController.addListener(() {
      if (mounted) {
        setState(() {});
      }
    });

    try{
      await cameraController.initialize();
      setState(() {});
    }
    catch(e) {
      if (e is CameraException) {
        switch (e.code) {
          case 'CameraAccessDenied':
            setState(() {
              isAccessDenied = true;
            });
            break;
          default:
          // Handle other errors here.
            break;
        }
      }
    }
  }

  @override
  void dispose() {
    controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final CameraController? cameraController = controller;

    if (cameraController == null || !cameraController.value.isInitialized) {
      return Container();
    }
    else if (isAccessDenied){
      return const Text("Please go to settings and allow camera access.");
    }

    return
      Column(
        children: [
          SizedBox(
            width: MediaQuery.of(context).size.height / 3,
            height: MediaQuery.of(context).size.height / 3,
            child: ClipRect(
              child: OverflowBox(
                alignment: Alignment.center,
                child: FittedBox(
                  fit: BoxFit.fitWidth,
                  child: SizedBox(
                    width: MediaQuery.of(context).size.height / 3,
                    height: MediaQuery.of(context).size.height / 3,
                    child: CameraPreview(cameraController),
                  ),
                ),
              ),
            ),
          ),
          // Expanded(child: CameraPreview(cameraController)),
          // Expanded(child: CameraPreview(controller)),
          ElevatedButton(
            onPressed: () async {
              try {
                // take picture and run callback on it
                // lock focus and exposure
                await cameraController.setFocusMode(FocusMode.locked);
                await cameraController.setExposureMode(ExposureMode.locked);
                final image = await cameraController.takePicture();
                widget.callback(File(image.path));
                // unlock
                await cameraController.setFocusMode(FocusMode.auto);
                await cameraController.setExposureMode(ExposureMode.auto);

              } catch (e) {
                print(e);
              }
            },
            child: const Icon(Icons.camera_alt),
          )
      ]
    );
  }
}