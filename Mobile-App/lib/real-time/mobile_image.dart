import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import "package:vector_math/vector_math_64.dart" as vmath;

class MobileImage extends StatefulWidget {
  final Uint8List _image;
  const MobileImage(Uint8List image, {super.key}): _image = image;

  @override
  State<MobileImage> createState() => _MobileImageState();
}

class _MobileImageState extends State<MobileImage> {
  double _prevScale = 1;
  double _prevRotation = 0;

  double _scale = 1;
  double _rotation = 0;
  (double, double) _translation = (0, 0);

  @override
  Widget build(BuildContext context) {
    // This method is rerun every time setState is called, for instance as done
    return Scaffold(
        body: GestureDetector(
            onScaleUpdate: (scaleUpdateDetails){
              // when the user makes a gesture update the currently viewed scale and rotation
              setState(() {
                // translation is prev translation + translation since last update
                _translation = (
                _translation.$1 + scaleUpdateDetails.focalPointDelta.dx,
                _translation.$2 + scaleUpdateDetails.focalPointDelta.dy
                );

                // scale is old scale * scale since start
                _scale = _prevScale * scaleUpdateDetails.scale;

                // rotation is old rotation + rotation since start
                _rotation = _prevRotation + scaleUpdateDetails.rotation;

              });
            },
            onScaleEnd: (scaleEndDetails){
              // store the images current location scale and rotation data for next time
              _prevScale = _scale;
              _prevRotation = _rotation;
            },
            child: Center(
                child: Transform(
                    alignment: Alignment.center,
                    transform: vmath.Matrix4.compose(
                        vmath.Vector3(_translation.$1, _translation.$2, 0),
                        // rotate on z axis
                        vmath.Quaternion.axisAngle(vmath.Vector3(0, 0, 1), _rotation),
                        // scale all axis equally
                        vmath.Vector3.all(_scale)),
                    child: Image.memory(widget._image)
                )

            )
        )
    );
  }
}
