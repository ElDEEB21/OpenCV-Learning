# OpenCV Bootcamp Course

This folder contains all the code, projects, and notebooks from a comprehensive OpenCV bootcamp covering advanced computer vision techniques and applications.

## Course Information
- **Title**: OpenCV Bootcamp - Advanced Computer Vision with Python
- **Status**: ✅ Completed
- **Language**: Python
- **Format**: Jupyter Notebooks + Python Scripts
- **Focus**: Practical Computer Vision Applications

## Folder Structure

```
02-OpenCV-Bootcamp/
├── src/                           # Python source files
│   ├── Camera.py                  # Basic camera capture and display
│   ├── CameraFilters.py          # Real-time camera filters
│   ├── display_image.py          # Image display examples
│   ├── faceDetection.py          # DNN-based face detection
│   ├── realTimeObjectDetection.py # Real-time object detection
│   └── realtime_pose_estimation.py # Human pose estimation
├── notebooks/                     # Jupyter notebooks
│   ├── Getting_Started_with_Images.ipynb        # Image basics
│   ├── Basic_Image_Manipulations.ipynb         # Image operations
│   ├── Basic_Image_Enhancement_Using_Mathematical_Operations.ipynb
│   ├── Annotating_Images.ipynb                 # Drawing and annotations
│   ├── High_Dynamic_Range.ipynb                # HDR imaging
│   ├── Image_Alignment.ipynb                   # Image registration
│   ├── Panorama.ipynb                          # Panoramic imaging
│   ├── Object_Detection.ipynb                  # Object detection techniques
│   ├── Object_Tracking.ipynb                   # Object tracking algorithms
│   ├── Pose_Estimation.ipynb                   # Human pose estimation
│   └── Writing_a_video_using_OpenCV.ipynb      # Video writing
├── Models/                        # Pre-trained models and configs
│   ├── deploy.prototxt           # Face detection model architecture
│   ├── res10_300x300_ssd_iter_140000_fp16.caffemodel # Face detection weights
│   ├── pose_deploy_linevec_faster_4_stages.prototxt  # Pose estimation model
│   ├── ssd_mobilenet_v2_coco_2018_03_29.pbtxt       # Object detection config
│   └── coco_class_labels.txt     # COCO dataset class labels
├── Photos/                        # Sample images for testing
│   ├── checkerboard_*.png         # Calibration patterns
│   ├── New_Zealand_*.jpg          # Scenic images for processing
│   ├── Tiger_Woods*.png           # Sports images
│   ├── boat/                      # Image sequences
│   │   ├── boat1.jpg
│   │   ├── boat2.jpg
│   │   └── ...
│   └── HDR/                       # High Dynamic Range images
│       ├── img_0.033.jpg
│       ├── img_0.25.jpg
│       └── ...
├── Videos/                        # Sample videos for testing
│   ├── race_car.mp4              # Original tracking video
│   ├── race_car-CSRT.mp4         # CSRT tracking result
│   ├── race_car-KCF.mp4          # KCF tracking result
│   └── race_car_out.avi          # Output video
└── README.md                      # This file
```

## Topics Covered

### Core Computer Vision Concepts
- [x] Image loading, display, and basic manipulations
- [x] Mathematical operations on images
- [x] Image enhancement techniques
- [x] Drawing and annotation on images
- [x] Camera interfacing and real-time processing

### Advanced Image Processing
- [x] High Dynamic Range (HDR) imaging
- [x] Image alignment and registration
- [x] Panoramic image stitching
- [x] Feature detection and matching

### Deep Learning Applications
- [x] DNN-based face detection using Caffe models
- [x] Real-time object detection with SSD MobileNet
- [x] Human pose estimation using OpenPose
- [x] Real-time video processing

### Video Processing
- [x] Object tracking algorithms (CSRT, KCF)
- [x] Video writing and encoding
- [x] Real-time video analysis

## Key Learning Outcomes

1. **Advanced Image Processing**: HDR imaging, panorama creation, and image alignment
2. **Deep Learning Integration**: Using pre-trained models for detection and estimation
3. **Real-time Applications**: Camera integration and live video processing
4. **Object Detection & Tracking**: Modern techniques for object recognition and tracking
5. **Human Pose Estimation**: Real-time human pose detection and analysis

## Prerequisites

### Required Dependencies
```bash
pip install opencv-python numpy matplotlib
pip install opencv-contrib-python  # For additional algorithms
```

### Optional Dependencies (for enhanced functionality)
```bash
pip install jupyter notebook  # For running notebooks
pip install scipy            # For advanced image processing
```

## How to Run

### Python Scripts
Navigate to the project root and run scripts:
```bash
# Basic camera capture
python src/Camera.py

# Face detection with webcam
python src/faceDetection.py

# Real-time object detection
python src/realTimeObjectDetection.py

# Pose estimation
python src/realtime_pose_estimation.py
```

### Jupyter Notebooks
Start Jupyter and navigate to the notebooks directory:
```bash
jupyter notebook
# Open any .ipynb file from the notebooks/ directory
# Note: All image paths have been updated to use relative paths (../Photos/)
```

## Model Files

### Face Detection
- **Architecture**: `Models/deploy.prototxt`
- **Weights**: `Models/res10_300x300_ssd_iter_140000_fp16.caffemodel`
- **Framework**: Caffe
- **Purpose**: Real-time face detection

### Object Detection
- **Config**: `Models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt`
- **Labels**: `Models/coco_class_labels.txt`
- **Framework**: TensorFlow
- **Purpose**: Real-time object detection (80 COCO classes)

### Pose Estimation
- **Architecture**: `Models/pose_deploy_linevec_faster_4_stages.prototxt`
- **Framework**: Caffe (OpenPose)
- **Purpose**: Human pose keypoint detection

## Technical Notes

### Camera Usage
- Scripts use `cv2.VideoCapture(0)` for default camera
- Pass camera index as command line argument: `python src/Camera.py 1`

### Path Management
- All scripts use relative paths from project root
- Models and images are referenced correctly after reorganization
- Scripts automatically detect the correct paths

### Performance Considerations
- Face detection runs at ~30 FPS on modern hardware
- Object detection performance depends on input resolution
- Pose estimation is computationally intensive

## Troubleshooting

### Common Issues
1. **Camera not found**: Check camera permissions and availability
2. **Model files missing**: Ensure all model files are in `Models/` directory
3. **Import errors**: Install required dependencies with pip

### Model File Locations
If you encounter model loading errors, verify these files exist:
- `Models/deploy.prototxt`
- `Models/res10_300x300_ssd_iter_140000_fp16.caffemodel`
- `Models/coco_class_labels.txt`

## Example Applications

### Real-time Face Detection
```python
python src/faceDetection.py
# Press 'q' to quit, 's' to save current frame
```

### Object Detection
```python
python src/realTimeObjectDetection.py
# Detects 80 different object classes from COCO dataset
```

### Pose Estimation
```python
python src/realtime_pose_estimation.py
# Detects human pose keypoints in real-time
```

## Learning Path Recommendation

1. **Start with notebooks**: Begin with `Getting_Started_with_Images.ipynb`
2. **Progress through basics**: Complete image manipulation notebooks
3. **Explore advanced topics**: HDR, panorama, and alignment
4. **Try real-time applications**: Run camera-based Python scripts
5. **Experiment with models**: Modify detection thresholds and parameters

---

**Note**: This bootcamp builds upon fundamental OpenCV concepts and focuses on practical, real-world applications of computer vision techniques.
