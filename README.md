# OpenCV Learning Journey 📸🤖

Welcome to my OpenCV learning repository! This repo documents my journey learning computer vision with OpenCV, starting from the basics and progressing to more advanced concepts.

## About This Repository

This repository serves as my personal learning log and practice space for OpenCV (Open Source Computer Vision Library). I'm documenting my progress, code examples, and projects as I work through various tutorials and resources.

## Learning Resources

### ✅ Completed Resources
🎯 **FreeCodeCamp OpenCV Course**: [OpenCV Course - Full Tutorial with Python](https://youtu.be/oXlwWbU8l2o?si=sGK615i9zXabRJ4k)
- **Source**: FreeCodeCamp
- **Language**: Python
- **Status**: ✅ **COMPLETED**
- **Duration**: Comprehensive tutorial series covering fundamentals

🎓 **OpenCV Bootcamp (Free OpenCV Course)**: [Free OpenCV Course](https://opencv.org/university/free-opencv-course/)
- **Source**: OpenCV.org University
- **Alternative Link**: [YouTube Playlist](https://www.youtube.com/watch?v=hZWgEPOVnuM&list=PL6e-Bu0cqf_jyhItJm_hEAopg8XNvoMXY)
- **Course Platform**: [courses.opencv.org](https://courses.opencv.org)
- **Language**: Python
- **Status**: ✅ **COMPLETED**
- **Duration**: ~3 hours of video content + hands-on projects
- **Format**: 
  - Self-paced video modules
  - Integrated quizzes and Colab notebooks
  - Real-time applications and advanced projects
  - Official certification upon completion
- **Focus**: Advanced computer vision applications, deep learning integration, and real-time processing

### Additional Resources
📚 Other learning resources:
- Official OpenCV Documentation
- Computer Vision books and papers
- Practice projects and challenges

## Repository Structure

```
📁 OpenCV-Learning/
├── 📄 README.md                    # This file
├── 📁 01-FreeCodeCamp-Course/      # ✅ Completed FreeCodeCamp tutorial materials
│   ├── 📄 README.md               # Course-specific documentation
│   ├── 📄 Simpsons.ipynb          # Face recognition project notebook
│   ├── 📁 src/                    # Python practice files from the course
│   ├── 📁 Photos/                 # Sample images used in exercises
│   ├── 📁 Videos/                 # Sample videos for practice
│   └── 📁 models/                 # Trained models and data files
└── 📁 02-OpenCV-Bootcamp/          # ✅ Completed: OpenCV Bootcamp course materials
    ├── 📄 README.md               # Bootcamp-specific documentation
    ├── 📁 src/                    # Python scripts for real-time applications
    │   ├── Camera.py              # Basic camera capture
    │   ├── faceDetection.py       # DNN-based face detection
    │   ├── realTimeObjectDetection.py # Object detection
    │   └── realtime_pose_estimation.py # Human pose estimation
    ├── 📁 notebooks/              # Jupyter notebooks for learning
    │   ├── Getting_Started_with_Images.ipynb
    │   ├── Basic_Image_Manipulations.ipynb
    │   ├── Object_Detection.ipynb
    │   └── ... (more notebooks)
    ├── 📁 Models/                 # Pre-trained models and configs
    │   ├── deploy.prototxt        # Face detection model
    │   ├── res10_300x300_ssd_iter_140000_fp16.caffemodel
    │   └── coco_class_labels.txt  # Object class labels
    ├── 📁 Photos/                 # Sample images for bootcamp exercises
    └── 📁 Videos/                 # Sample videos for tracking/analysis
```

## Learning Progress

### ✅ Completed
**FreeCodeCamp OpenCV Course**:
- [x] Basic OpenCV setup and fundamentals
- [x] Image and video handling
- [x] Image filtering and transformations
- [x] Color space conversions
- [x] Edge detection and contours
- [x] Face detection and recognition
- [x] Feature detection
- [x] Histogram analysis
- [x] Bitwise operations and masking
- [x] Image smoothing and gradients
- [x] Thresholding techniques
- [x] Geometric transformations

**OpenCV Bootcamp (Advanced Course)**:
- [x] **Getting Started with Images** – Image loading, display, and basic operations
- [x] **Basic Image Manipulations** – Resizing, cropping, mathematical operations
- [x] **Image Enhancement** – Histogram equalization and mathematical transformations
- [x] **Annotating Images** – Drawing shapes, text, and annotations
- [x] **High Dynamic Range (HDR)** – Multi-exposure image processing
- [x] **Image Alignment** – Registration and geometric transformations
- [x] **Panoramic Imaging** – Stitching multiple images together
- [x] **Object Detection** – DNN-based detection using pre-trained models
- [x] **Object Tracking** – Real-time tracking algorithms (CSRT, KCF)
- [x] **Human Pose Estimation** – Real-time pose keypoint detection
- [x] **Video Processing** – Reading, writing, and analyzing video streams
- [x] **Real-time Applications** – Camera integration and live processing

**Advanced Features Mastered:**
- ✅ **DNN Face Detection** – Using Caffe models for real-time face detection
- ✅ **Real-time Object Detection** – SSD MobileNet for 80 COCO classes
- ✅ **Human Pose Estimation** – OpenPose integration for keypoint detection
- ✅ **Camera Integration** – Live video processing and filtering
- ✅ **Model Integration** – Working with pre-trained deep learning models
- ✅ **Performance Optimization** – Real-time processing techniques
- ✅ **Advanced Image Processing** – HDR, panorama stitching, image alignment

## Prerequisites

- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- Matplotlib (for displaying images)
- Jupyter Notebook (for running bootcamp notebooks)

## Installation

```bash
# Core OpenCV installation
pip install opencv-python

# Additional dependencies for all courses
pip install numpy matplotlib

# For bootcamp advanced features
pip install opencv-contrib-python

# For running Jupyter notebooks
pip install jupyter notebook
```

## Usage

### FreeCodeCamp Course Materials (Completed)
The Python files in the `01-FreeCodeCamp-Course/src/` directory demonstrate various OpenCV concepts from the completed tutorial series. Each file focuses on specific functionality.

```bash
# Navigate to the FreeCodeCamp course directory
cd "01-FreeCodeCamp-Course/src"

# Run any of the practice files
python <filename>.py
```

### OpenCV Bootcamp Course (Completed)
The `02-OpenCV-Bootcamp/` directory contains organized materials from the completed OpenCV bootcamp:

**Running Python Scripts:**
```bash
# Navigate to project root and run scripts
cd "02-OpenCV-Bootcamp"

# Basic camera capture
python src/Camera.py

# Real-time face detection
python src/faceDetection.py

# Object detection (80 COCO classes)
python src/realTimeObjectDetection.py

# Human pose estimation
python src/realtime_pose_estimation.py
```

**Using Jupyter Notebooks:**
```bash
# Start Jupyter in the bootcamp directory
cd "02-OpenCV-Bootcamp"
jupyter notebook

# Open any notebook from the notebooks/ directory
```

**Key Features Available:**
- 📹 Real-time camera processing with multiple algorithms
- 🧠 Deep learning model integration (Caffe, TensorFlow)
- 🎯 Pre-trained models for face detection, object detection, and pose estimation
- 📊 Performance-optimized implementations
- 🔧 Organized codebase with proper path management

## Notes and Tips

💡 **Learning Notes**:
- Always check image paths and ensure files exist
- Remember that OpenCV uses BGR color format by default
- Practice with different image types and sizes
- Experiment with parameters to understand their effects
- Real-time applications require performance optimization
- Deep learning models need proper preprocessing and post-processing
- Always consider lighting conditions and camera calibration for robust applications

## Future Plans

- ✅ ~~Complete FreeCodeCamp OpenCV fundamentals course~~
- ✅ ~~Complete OpenCV University advanced bootcamp course~~
- 🎯 **Next Steps**:
  - Explore advanced deep learning frameworks integration (PyTorch, TensorFlow)
  - Build custom computer vision applications and projects
  - Investigate YOLO and other modern object detection architectures
  - Develop mobile computer vision applications
  - Create a comprehensive computer vision portfolio
  - Explore computer vision in specialized domains (medical imaging, autonomous vehicles, etc.)
  - Contribute to open-source computer vision projects

## Achievements & Certifications

- 🏆 **FreeCodeCamp OpenCV Course** - Completed (June 2025)
- 🏆 **OpenCV University Bootcamp** - Completed (June 2025)
- 📜 **Skills Mastered**: 
  - Core OpenCV operations and image processing
  - Real-time computer vision applications
  - Deep learning model integration
  - Object detection and tracking
  - Human pose estimation
  - Advanced image processing techniques (HDR, panorama, alignment)

## Contributing

This is a personal learning repository, but feel free to:
- Suggest improvements or corrections
- Share additional learning resources
- Provide feedback on code examples

---

📅 **Started**: June 2025  
🎯 **Goal**: Master computer vision fundamentals and advanced techniques with OpenCV  
📚 **Status**: Core Learning Complete - Moving to Advanced Projects  
🏆 **Achievements**: 2 Major Courses Completed, Multiple Real-time Applications Built