import 'dart:io';

import 'package:flutter/material.dart';
import 'package:live_traffic_sign_classification/camera.dart';
import 'package:live_traffic_sign_classification/image_processor.dart';

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
  // final Camera _camera = ;
  List<dynamic>? _prediction;

  _MyHomePageState(){
    print("HALELLOO????");
    ImageProcessor.create("assets/processing/yolov5s.torchscript").then((processor) {
      processor.run(File("assets/processing/zidane.jpg")).then((prediction) {
        print("THIS IS HAPPENING");
        _prediction = prediction;
        setState(() {});
      });
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
            const Text(
              'Camera:',
            ),
            Container(
              margin: const EdgeInsets.all(10.0),
              height: MediaQuery.of(context).size.height / 3,
              child: const Camera()
            ),
            Text(
              'The prediction is: $_prediction',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
          ],
        ),
      ),
    );
  }
}
