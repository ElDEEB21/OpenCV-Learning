# FreeCodeCamp OpenCV Course

This folder contains all the code and projects from the comprehensive OpenCV tutorial by FreeCodeCamp.

## Course Information
- **Title**: OpenCV Course - Full Tutorial with Python
- **Source**: [FreeCodeCamp YouTube](https://youtu.be/oXlwWbU8l2o?si=sGK615i9zXabRJ4k)
- **Status**: ✅ Completed
- **Duration**: ~3.5 hours
- **Language**: Python

## Folder Structure

```
01-FreeCodeCamp-Course/
├── src/                    # Python source files
│   ├── basics.py          # Basic OpenCV operations
│   ├── reader.py          # Image and video reading
│   ├── rescale.py         # Image resizing and rescaling
│   ├── draw.py            # Drawing shapes and text
│   ├── spaces.py          # Color space conversions
│   ├── splitmerge.py      # Color channel operations
│   ├── thresh.py          # Thresholding techniques
│   ├── transformations.py # Image transformations
│   ├── contours.py        # Contour detection
│   ├── masking.py         # Image masking
│   ├── histogram.py       # Histogram analysis
│   ├── gradients.py       # Image gradients
│   ├── bitwise.py         # Bitwise operations
│   ├── smoothing.py       # Image smoothing/filtering
│   ├── face_detect.py     # Face detection
│   ├── faces_train.py     # Face recognition training
│   └── face_recognition.py # Face recognition implementation
├── models/                # Trained models and data
│   ├── face_trained.yml   # Trained face recognition model
│   ├── features.npy       # Feature data
│   ├── labels.npy         # Label data
│   └── haar_cascade.xml   # Haar cascade classifier
├── Simpsons.ipynb        # Jupyter notebook examples
└── README.md             # This file
```

## Topics Covered

### Core Concepts
- [x] Reading and displaying images/videos
- [x] Image resizing and rescaling
- [x] Drawing shapes and text on images
- [x] Color space conversions (BGR, RGB, HSV, LAB)
- [x] Color channel splitting and merging

### Image Processing
- [x] Thresholding (Binary, Adaptive)
- [x] Image transformations (Translation, Rotation, Flipping)
- [x] Contour detection and analysis
- [x] Image masking and bitwise operations
- [x] Histogram computation and analysis
- [x] Gradient and edge detection
- [x] Image smoothing and filtering

### Advanced Features
- [x] Face detection using Haar Cascades
- [x] Face recognition system implementation
- [x] Training custom face recognition models
- [x] Real-time video processing

## Key Learning Outcomes

1. **Image Fundamentals**: Understanding image representation, color spaces, and basic operations
2. **Image Processing**: Applying filters, transformations, and enhancement techniques
3. **Feature Detection**: Identifying and extracting features from images
4. **Object Detection**: Implementing face detection and recognition systems
5. **Real-time Processing**: Working with video streams and live camera feeds

## How to Run

Make sure you have the required dependencies:
```bash
pip install opencv-python numpy matplotlib
```

Run any script from the src directory:
```bash
python src/basics.py
python src/face_detect.py
# etc.
```

## Notes
- All image paths in the source files reference `../resources/Photos/`
- Video files are located in `../resources/Videos/`
- Model files are saved in the `models/` directory
- Some scripts may need path adjustments after the reorganization
