import 'dart:ffi';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:live_traffic_sign_classification/camera.dart';
import 'package:live_traffic_sign_classification/image_processor.dart';


import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {

    return MaterialApp(
      title: 'BBB Traffic Sign Detector',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Traffic Sign Detector'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  // Fields in a Widget subclass are always marked "final".
  // immutable state
  final String title;

  // State object in "StatefulWidget" class
  // holds all mutable state
  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late Camera _camera;
  ImageProcessor? _processor;
  List<dynamic>? _prediction;
  Widget picture = Container();

  void pictureTaken(File newPicture){
    setState((){
      // getImageFileFromAssets("processing/zidane.jpg").then((newPicture){
          _processor?.run(newPicture).then((prediction) {
            print("THIS IS HAPPENING");
            picture = prediction;
            setState(() {});
          });
      // });
    });
  }

  Future<File> getImageFileFromAssets(String path) async {
    final byteData = await rootBundle.load('assets/$path');

    final file = File('${(await getTemporaryDirectory()).path}/$path');
    await file.create(recursive: true);
    await file.writeAsBytes(byteData.buffer.asUint8List(byteData.offsetInBytes, byteData.lengthInBytes));

    return file;
  }


  _MyHomePageState(){
    _camera = Camera(callback: pictureTaken);
    ImageProcessor.create("assets/processing/yolov5s.torchscript").then((processor) => {
      setState((){
        _processor = processor;
      })
    });
  }

  // This method is rerun every time setState is called, for instance as done
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        // TRY THIS: Try changing the color here to a specific color (to
        // Colors.amber, perhaps?) and trigger a hot reload to see the AppBar
        // change color while the other colors stay the same.
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        // Here we take the value from the MyHomePage object that was created by
        // the App.build method, and use it to set our appbar title.
        title: Text(widget.title),
      ),
      body: Center(
        // Center is a layout widget. It takes a single child and positions it
        // in the middle of the parent.
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _camera,
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
                      child: picture,
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
