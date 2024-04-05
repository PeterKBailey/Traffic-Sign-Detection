import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'mobile_image.dart';

class Gallery extends StatefulWidget{
  final List<Uint8List> _images;
  const Gallery(List<Uint8List> images, {super.key}): _images = images;

  @override
  State<StatefulWidget> createState() {
    return _GalleryState();
  }
}

class _GalleryState extends State<Gallery>{
  _GalleryState();
  @override
  Widget build(BuildContext context) {
    print(widget._images.length);
    return GridView.count(
      crossAxisCount: 4,
      children: [...widget._images.map((image){
        return ImageCard(image);
      })],
    );
  }
}

class ImageCard extends StatelessWidget{
  final Uint8List _image;
  const ImageCard(Uint8List image, {super.key}): _image = image;

  @override
  Widget build(BuildContext context) {
    return
      GestureDetector(
        onTap: (){
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context){
              return MobileImage(_image);
            }),
          );
        },
        child: Padding(
          padding: const EdgeInsets.all(3),
          child: FittedBox(
            fit: BoxFit.cover,
            clipBehavior: Clip.hardEdge,
            child:Image.memory(_image)
          )
        )
      );
  }
}
