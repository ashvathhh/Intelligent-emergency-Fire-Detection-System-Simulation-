import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ffmpeg_kit_flutter/ffmpeg_kit.dart';
import 'package:video_player/video_player.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  runApp(MyApp());
}

/// Main app widget with dark theme and HomeScreen as the landing page.
class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'RESCUE',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: HomeScreen(),
    );
  }
}

/// HomeScreen: Shows title and an upload button.
class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              "RESCUE",
              style: TextStyle(
                color: Colors.white,
                fontSize: 36,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 50),
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => VideoProcessingPage()),
                );
              },
              style: ElevatedButton.styleFrom(
                shape: CircleBorder(),
                padding: EdgeInsets.all(24),
                backgroundColor: Colors.red,
              ),
              child: Icon(Icons.upload, size: 32, color: Colors.white),
            ),
          ],
        ),
      ),
    );
  }
}

/// VideoProcessingPage: Lets the user upload a video, processes frames, and overlays detections.
class VideoProcessingPage extends StatefulWidget {
  @override
  _VideoProcessingPageState createState() => _VideoProcessingPageState();
}

class _VideoProcessingPageState extends State<VideoProcessingPage> {
  VideoPlayerController? _videoController;
  String? _videoPath;
  List<Map<String, dynamic>> _detections = [];
  double _originalWidth = 0;
  double _originalHeight = 0;
  final int _inputSize = 640;
  // Confidence threshold for filtering detections.
  final double _confidenceThreshold = 0.35;
  Uint8List? _modelBytes;
  Interpreter? _interpreter;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  /// Loads the TFLite model from assets and creates a persistent interpreter.
  Future<void> _loadModel() async {
    try {
      final modelData = await rootBundle.load('assets/best_float32.tflite');
      _modelBytes = modelData.buffer.asUint8List();
      _interpreter = await Interpreter.fromBuffer(_modelBytes!);
      _interpreter!.allocateTensors();
      print("✅ Model loaded successfully");
    } catch (e) {
      print("❌ Error loading model: $e");
    }
  }

  /// Allows the user to pick a video from the gallery.
  Future<void> _pickVideo() async {
    final pickedFile = await ImagePicker().pickVideo(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _videoPath = pickedFile.path;
        _videoController = VideoPlayerController.file(File(_videoPath!))
          ..initialize().then((_) {
            setState(() {
              _originalWidth = _videoController!.value.size.width;
              _originalHeight = _videoController!.value.size.height;
            });
            _videoController!.play();
          });
      });
    }
  }

  /// Processes the video:
  /// 1. Extracts frames via FFmpeg (scaling them to 640x640).
  /// 2. Processes each frame for detection.
  Future<void> _processVideo() async {
    if (_videoPath == null || _modelBytes == null || _interpreter == null) return;

    final Directory dir = await getTemporaryDirectory();
    final Directory frameDir = Directory('${dir.path}/frames');
    if (!frameDir.existsSync()) frameDir.createSync(recursive: true);

    // FFmpeg command: Scale frames to 640x640 and ensure proper color space conversion
    await FFmpegKit.execute(
      '-i $_videoPath -vf "scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2:black" -q:v 2 ${frameDir.path}/frame_%03d.jpg'
    );

    final List<File> frameFiles =
        frameDir.listSync().whereType<File>().toList();
    print("Extracted ${frameFiles.length} frames");

    // Process frames in smaller batches for faster processing
    final int batchSize = 3;
    for (var i = 0; i < frameFiles.length; i += batchSize) {
      final batch = frameFiles.skip(i).take(batchSize);
      final futures = batch.map((frameFile) {
        final Uint8List frameData = frameFile.readAsBytesSync();
        return compute(
          _processFrameInIsolate,
          IsolateProcessingInput(
            frameData,
            _inputSize,
            _originalWidth,
            _originalHeight,
            _modelBytes!,
            _confidenceThreshold,
          ),
        );
      });
      final batchResults = await Future.wait(futures);
      setState(() {
        _detections = batchResults.expand((x) => x).toList();
      });
    }
    print("🎉 Processing Complete!");
  }

  @override
  void dispose() {
    _videoController?.dispose();
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: Text("Thermal Human Detection"),
        backgroundColor: Colors.black,
      ),
      body: Column(
        children: [
          if (_videoController != null && _videoController!.value.isInitialized)
            Expanded(
              child: AspectRatio(
                aspectRatio: _videoController!.value.aspectRatio,
                child: Stack(
                  children: [
                    VideoPlayer(_videoController!),
                    CustomPaint(
                      painter: BoundingBoxPainter(
                        _detections,
                        _originalWidth,
                        _originalHeight,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          Padding(
            padding: EdgeInsets.all(16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton(
                  onPressed: _pickVideo,
                  child: Text("📂 Upload Video"),
                ),
                ElevatedButton(
                  onPressed: _processVideo,
                  child: Text("🚀 Start Detection"),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// Structure for passing data to the isolate.
class IsolateProcessingInput {
  final Uint8List frameData;
  final int inputSize;
  final double originalWidth;
  final double originalHeight;
  final Uint8List modelBytes;
  final double confidenceThreshold;
  IsolateProcessingInput(
    this.frameData,
    this.inputSize,
    this.originalWidth,
    this.originalHeight,
    this.modelBytes,
    this.confidenceThreshold,
  );
}

/// This function runs in a background isolate.
/// It decodes the frame, resizes to 640x640, converts it to a flat Float32List (normalized to [0,1]),
/// runs inference with the TFLite model, and parses detections.
Future<List<Map<String, dynamic>>> _processFrameInIsolate(IsolateProcessingInput input) async {
  try {
    final img.Image? image = img.decodeImage(input.frameData);
    if (image == null) return [];
    final img.Image resizedImage = img.copyResize(image, width: input.inputSize, height: input.inputSize);

    final int tensorSize = input.inputSize * input.inputSize * 3;
    final Float32List inputBuffer = Float32List(tensorSize);
    int index = 0;
    final pixels = resizedImage.getBytes();
    for (int i = 0; i < pixels.length; i += 4) {
      // Normalize pixel values to [0,1] and ensure RGB order
      inputBuffer[index] = pixels[i].toDouble() / 255.0; // R
      inputBuffer[index + 1] = pixels[i + 1].toDouble() / 255.0; // G
      inputBuffer[index + 2] = pixels[i + 2].toDouble() / 255.0; // B
      index += 3;
    }
    // Verify input buffer is properly filled
    if (index != tensorSize) {
      throw Exception("Input buffer not completely filled: $index vs $tensorSize");
    }

    // Prepare input and output tensors
    final int numBoxes = 8400;
    
    // Create input tensor array with shape [1, 640, 640, 3]
    final inputArray = List.generate(1, (_) => List.generate(input.inputSize, (_) => List.generate(input.inputSize, (_) => List<double>.filled(3, 0.0))));
    for (int y = 0; y < input.inputSize; y++) {
      for (int x = 0; x < input.inputSize; x++) {
        for (int c = 0; c < 3; c++) {
          inputArray[0][y][x][c] = inputBuffer[y * input.inputSize * 3 + x * 3 + c];
        }
      }
    }
    
    // Initialize output array with optimized shape [1, 5, 8400]
    final outputArray = List.generate(1, (_) => List.generate(5, (_) => List<double>.filled(numBoxes, 0.0)));

    // Run inference using the persistent interpreter with optimized memory allocation
    final interpreter = await Interpreter.fromBuffer(input.modelBytes, options: InterpreterOptions()..threads = 4);
    interpreter.allocateTensors();

    try {
      final inputTensor = interpreter.getInputTensor(0);
      final outputTensor = interpreter.getOutputTensor(0);
      
      if (inputTensor.shape[1] != input.inputSize || inputTensor.shape[2] != input.inputSize) {
        print("Input tensor shape mismatch: expected [1,${input.inputSize},${input.inputSize},3], got ${inputTensor.shape}");
        interpreter.close();
        return [];
      }

      // Run inference with properly formatted input
      interpreter.run(inputArray, outputArray);
      print("✅ Inference completed successfully");

      // Print output tensor shape for debugging
      print("Output tensor shape: ${outputTensor.shape}");
      print("Output array shape: [${outputArray.length}, ${outputArray[0].length}, ${outputArray[0][0].length}]");

      // Verify output tensor shape matches expected [1, 5, 8400]
      if (outputTensor.shape[0] != 1 || outputTensor.shape[1] != 5 || outputTensor.shape[2] != numBoxes) {
        print("Output tensor shape mismatch: expected [1,5,$numBoxes], got ${outputTensor.shape}");
        interpreter.close();
        return [];
      }

      // Convert outputArray to List<List<List<double>>> before passing to _parseDetections
      final List<List<List<double>>> convertedOutput = [
        outputArray[0].map((list) => List<double>.from(list)).toList()
      ];

      // Parse detections from the output tensor [1, 5, 8400]
      return _parseDetections(
        convertedOutput,
        input.inputSize,
        input.originalWidth,
        input.originalHeight,
        input.confidenceThreshold
      );
    } catch (e) {
      print("Inference error: $e");
      interpreter.close();
      return [];
    } finally {
      interpreter.close();
    }

  } catch (e) {
    print("Error processing frame: $e");
    return [];
  }




}

/// Parses the model output (shape [1, 5, 8400]) into a list of detection maps.
/// Converts normalized outputs (relative to 640) to pixel coordinates in the original video.
List<Map<String, dynamic>> _parseDetections(
    List<List<List<double>>> output,
    int inputSize,
    double originalWidth,
    double originalHeight,
    double confidenceThreshold) {
  final List<Map<String, dynamic>> detections = [];
  final int numBoxes = output[0][0].length;
  // Calculate aspect ratio preserving scale factors
  final double scale = min(originalWidth / inputSize, originalHeight / inputSize);
  final double offsetX = (originalWidth - inputSize * scale) / 2;
  final double offsetY = (originalHeight - inputSize * scale) / 2;

  print("📊 Processing ${numBoxes} potential detections...");

  // First pass: collect all valid detections
  List<Map<String, dynamic>> validDetections = [];
  for (int i = 0; i < numBoxes; i++) {
    try {
      final double confidence = output[0][4][i];
      if (confidence < 0.25) continue;  // Lower threshold for better detection rate

      // Get normalized coordinates (0-1 range) with validation
      final double xCenter = output[0][0][i].clamp(0.0, 1.0);
      final double yCenter = output[0][1][i].clamp(0.0, 1.0);
      final double boxWidth = output[0][2][i].clamp(0.0, 1.0);
      final double boxHeight = output[0][3][i].clamp(0.0, 1.0);

      if (xCenter.isNaN || yCenter.isNaN || boxWidth.isNaN || boxHeight.isNaN) {
        print("Invalid detection values at index $i");
        continue;
      }

      // Scale coordinates to original image dimensions while preserving aspect ratio
      final double finalXCenter = (xCenter * inputSize * scale + offsetX).clamp(0.0, originalWidth);
      final double finalYCenter = (yCenter * inputSize * scale + offsetY).clamp(0.0, originalHeight);
      final double finalWidth = (boxWidth * inputSize * scale).clamp(0.0, originalWidth);
      final double finalHeight = (boxHeight * inputSize * scale).clamp(0.0, originalHeight);

      validDetections.add({
        "xMin": finalXCenter - finalWidth / 2,
        "yMin": finalYCenter - finalHeight / 2,
        "width": finalWidth,
        "height": finalHeight,
        "confidence": confidence,
      });
    } catch (e) {
      print("Error processing detection at index $i: $e");
      continue;
    }
  }

  // Sort detections by confidence score (highest first)
  validDetections.sort((a, b) => b["confidence"].compareTo(a["confidence"]));

  // Apply Non-Maximum Suppression
  for (int i = 0; i < validDetections.length; i++) {
    if (validDetections[i]["confidence"] == 0) continue; // Skip if already suppressed

    final box1 = validDetections[i];
    detections.add(box1); // Add highest confidence detection

    // Compare with remaining boxes
    for (int j = i + 1; j < validDetections.length; j++) {
      final box2 = validDetections[j];
      if (box2["confidence"] == 0) continue; // Skip if already suppressed

      // Calculate IoU between boxes
      final double iou = _calculateIoU(box1, box2);
      if (iou > 0.45) { // IoU threshold
        validDetections[j]["confidence"] = 0; // Suppress overlapping box
      }
    }
  }

  print("✅ Found ${detections.length} detections after NMS");
  return detections;
}

/// Calculate Intersection over Union (IoU) between two bounding boxes
double _calculateIoU(Map<String, dynamic> box1, Map<String, dynamic> box2) {
  final double box1X1 = box1["xMin"];
  final double box1Y1 = box1["yMin"];
  final double box1X2 = box1["xMin"] + box1["width"];
  final double box1Y2 = box1["yMin"] + box1["height"];

  final double box2X1 = box2["xMin"];
  final double box2Y1 = box2["yMin"];
  final double box2X2 = box2["xMin"] + box2["width"];
  final double box2Y2 = box2["yMin"] + box2["height"];

  final double intersectionX1 = max(box1X1, box2X1);
  final double intersectionY1 = max(box1Y1, box2Y1);
  final double intersectionX2 = min(box1X2, box2X2);
  final double intersectionY2 = min(box1Y2, box2Y2);

  if (intersectionX2 <= intersectionX1 || intersectionY2 <= intersectionY1) {
    return 0.0;
  }

  final double intersectionArea = (intersectionX2 - intersectionX1) * (intersectionY2 - intersectionY1);
  final double box1Area = box1["width"] * box1["height"];
  final double box2Area = box2["width"] * box2["height"];

  return intersectionArea / (box1Area + box2Area - intersectionArea);
  }



/// CustomPainter to draw bounding boxes over the video.
class BoundingBoxPainter extends CustomPainter {
  final List<Map<String, dynamic>> detections;
  final double originalWidth;
  final double originalHeight;
  BoundingBoxPainter(this.detections, this.originalWidth, this.originalHeight);

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..color = Colors.green
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    // Draw each bounding box.
    for (final detection in detections) {
      final double xMin = detection["xMin"];
      final double yMin = detection["yMin"];
      final double width = detection["width"];
      final double height = detection["height"];

      canvas.drawRect(Rect.fromLTWH(xMin, yMin, width, height), paint);

      final TextPainter textPainter = TextPainter(
        text: TextSpan(
          text: "${(detection["confidence"] * 100).toStringAsFixed(1)}%",
          style: TextStyle(
            color: Colors.red,
            fontSize: 12,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout(minWidth: 0, maxWidth: size.width);
      textPainter.paint(canvas, Offset(xMin, yMin - 14));
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}