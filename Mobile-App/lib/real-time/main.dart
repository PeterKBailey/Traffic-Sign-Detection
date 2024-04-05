import 'dart:typed_data';
import 'package:flutter/material.dart';

import 'camera.dart';
import 'gallery.dart';

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
  final List<Uint8List> _images = List.empty(growable: true);

  late Camera _camera;
  int _navigationIndex = 0;

  _MyAppState(){
    _camera = Camera(imageHandler);
  }

  void imageHandler(Uint8List image){
    setState(() {
      _images.add(image);
    });
  }

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
          body: [
            _camera,
            Gallery(_images)
          ][_navigationIndex],
          bottomNavigationBar: NavigationBar(
            selectedIndex: _navigationIndex,
            destinations: const [
              NavigationDestination(
                icon: Icon(Icons.camera),
                label: "Detection"
              ),
              NavigationDestination(
                  icon: Icon(Icons.photo_library_sharp),
                  label: "Gallery"
              )
            ],
            onDestinationSelected: (int index){
              setState((){
                _navigationIndex = index;
              });
            },
          ),
          // floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
          // floatingActionButton: TakePictureButton(screenshotController: screenshotController,)
        )
    );
  }
}



