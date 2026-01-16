Human Detection System for Emergency Fire Evacuation



Real-time human detection system using YOLOv8 for emergency fire scenarios. Detects people in smoke-filled environments using thermal imaging with 92% accuracy.



Note: Originally designed for real-time detection but currently implemented for pre-recorded video analysis.



Features



\- Human detection with YOLOv8

\- Mobile app (Flutter) for video processing

\- Works in low visibility/smoke conditions

\- 6-8ms inference speed per frame

\- Offline capability with TFLite

\- Video upload and analysis



Quick Start



Clone and install:

git clone https://github.com/username/human-detection.git

cd human-detection

pip install -r requirements.txt



Train model:

python model/train.py --data config/data.yaml --epochs 50 --batch 16



Run mobile app:

cd mobile-app

flutter pub get

flutter run



Requirements



Training:

\- Python 3.8+

\- NVIDIA GPU (16GB VRAM)

\- 16GB RAM



Mobile:

\- Flutter 3.0+

\- Android 8.0+ or iOS 12.0+

\- 4GB RAM device



Dataset



Intruder-Thermal Dataset:

\- 2,000 training images (640x640)

\- 500 test images

\- JSON annotations with bounding boxes

\- Two-phase training: raw frames then preprocessed frames



Performance



Detection Metrics:

\- mAP@0.5: 89.6% (demonstrates strong overall detection)

\- Precision: 91% at standard threshold

\- Maximum Precision: 100% at confidence 0.941 (zero false positives)

\- Recall: 88% at standard threshold

\- Maximum Recall: 96% at confidence 0.0 (captures all potential positives)

\- F1 Score: 84% at optimal confidence 0.548

\- Inference Speed: 6-8ms per frame on mobile



Confusion Matrix:

\- True positives: 302

\- False positives: 103

\- False negatives: 38

\- Overall accuracy: 89%



Training Details:

\- 50 epochs

\- Two-phase approach: raw thermal images then preprocessed/normalized frames

\- Custom object detection loss function for bounding box regression and classification

\- Performance validated through precision-recall curves and loss visualizations



Project Structure



human-detection/

├── mobile-app/         Flutter app

├── model/              Training scripts

├── dataset/            Training data

├── configs/            Config files

└── outputs/            Results and performance curves



Configuration



Edit configs/config.yaml:

\- Model: YOLOv8

\- Input: 640x640

\- Batch size: 16

\- Confidence: 0.5 (standard), 0.548 (optimal F1)

\- Learning rate: 0.001



Model Details



Architecture:

\- Backbone: CSPDarknet53

\- Neck: PAN (Path Aggregation Network)

\- Head: YOLO detection head

\- Post-processing: NMS for redundant detection filtering



Training setup:

\- Phase 1: Raw thermal image frames

\- Phase 2: Preprocessed and normalized frames

\- Adam optimizer

\- Combined loss function (localization + classification)

\- Data augmentation applied



Dependencies



Main libraries:

ultralytics

tensorflow-lite

opencv-python

numpy

pandas

flutter (mobile)



Full list in requirements.txt



Usage



Process video:

python main.py --input video.mp4 --output result.mp4 --conf 0.5



Convert to TFLite:

python model/model\_conversion.py --weights best.pt --output best.tflite



Mobile app workflow:

1\. Launch app

2\. Press "Rescue" button

3\. Upload pre-recorded video

4\. System segments and resizes frames to 640x640

5\. YOLOv8 processes frames with CSPDarknet53 backbone

6\. NMS filters redundant detections

7\. View results with bounding boxes, class labels, and confidence scores



Current Limitations:

\- Real-time camera detection not yet implemented

\- Currently processes uploaded videos only

\- Live detection feature planned for future release



Files



Key files:

\- model/train.py - Training script

\- model/model\_conversion.py - TFLite conversion

\- mobile-app/lib/main.dart - App entry point

\- configs/data.yaml - Dataset config

\- outputs/ - Performance curves (precision-recall, F1-confidence, etc.)



Hardware Used



Training:

\- Intel Core i7

\- NVIDIA T4 GPU

\- 16GB RAM

\- 256GB SSD



Testing:

\- Android/iOS device

\- 4GB RAM

\- 1080p camera



Results



Model achieves optimal performance at confidence threshold 0.548 with balanced precision-recall tradeoff. 



Contact



Email: cheppash2@gmail.com



