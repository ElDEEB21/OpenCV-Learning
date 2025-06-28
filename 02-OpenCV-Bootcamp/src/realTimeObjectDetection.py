import cv2 as cv
import numpy as np
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level from src/

# Load COCO class labels
classFile = os.path.join(script_dir,'Models' ,"coco_class_labels.txt")
try:
    with open(classFile) as fp:
        labels = fp.read().strip().split("\n")
    print(f"Loaded {len(labels)} class labels")
except FileNotFoundError:
    print(f"Error: Could not find {classFile}")
    print("Please make sure the COCO class labels file exists in the same directory")
    sys.exit(1)

# Load the pre-trained SSD MobileNet v2 model trained on COCO dataset
modelFile = os.path.join(script_dir, "Models", "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb")
configFile = os.path.join(script_dir, "Models", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

try:
    # Read the TensorFlow network using OpenCV DNN module
    net = cv.dnn.readNetFromTensorflow(modelFile, configFile)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure the model files exist in the Models directory")
    sys.exit(1)

camera_source = 0
if len(sys.argv) > 1:
    camera_source = int(sys.argv[1])

cap = cv.VideoCapture(camera_source)
if not cap.isOpened():
    print(f"Error: Could not open camera {camera_source}")
    sys.exit(1)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FPS, 30)

window_name = 'Real-time Object Detection'
cv.namedWindow(window_name, cv.WINDOW_NORMAL)

# Model parameters for SSD MobileNet v2
input_width = 300
input_height = 300
mean_values = [0, 0, 0]  # No mean subtraction for this model
confidence_threshold = 0.5  # Minimum confidence for object detection
nms_threshold = 0.4

# Font settings for text display
font_face = cv.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1

def detect_objects(frame, net):
    """
    Perform object detection on a single frame using SSD MobileNet v2
    Returns detections array with bounding boxes, classes, and confidence scores
    """
    # Create a 4D blob from the frame for neural network input
    blob = cv.dnn.blobFromImage(
        frame, 
        scalefactor=1.0,
        size=(input_width, input_height),
        mean=mean_values,
        swapRB=True,  # Convert BGR to RGB
        crop=False
    )
    
    net.setInput(blob)
    # Forward pass returns detections: [batch_id, class_id, confidence, x1, y1, x2, y2]
    detections = net.forward()
    
    return detections

def draw_detection(frame, class_id, confidence, x, y, w, h):
    """
    Draw bounding box and label for detected object
    """
    label = f"{labels[class_id]} {confidence:.0%}"
    
    # Color palette for different object classes
    colors = [
        (255, 87, 34), (76, 175, 80), (33, 150, 243), (156, 39, 176), (255, 193, 7),
        (0, 188, 212), (244, 67, 54), (139, 195, 74), (63, 81, 181), (255, 152, 0),
    ]
    color = colors[class_id % len(colors)]
    
    cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    (text_width, text_height), baseline = cv.getTextSize(label, font_face, font_scale, font_thickness)
    
    # Draw label background
    padding = 4
    bg_x1, bg_y1 = x, y - text_height - baseline - (padding * 2)
    bg_x2, bg_y2 = x + text_width + (padding * 2), y
    
    overlay = frame.copy()
    cv.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, cv.FILLED)
    cv.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    cv.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
    
    cv.putText(frame, label, (x + padding, y - baseline - padding), 
               font_face, font_scale, (255, 255, 255), font_thickness, cv.LINE_AA)

def process_detections(frame, detections):
    """
    Process neural network output and draw bounding boxes for valid detections
    """
    height, width = frame.shape[:2]
    detection_count = 0
    
    for i in range(detections.shape[2]):
        class_id = int(detections[0, 0, i, 1])
        confidence = float(detections[0, 0, i, 2])
        
        # Only process detections above confidence threshold
        if confidence > confidence_threshold:
            # Convert normalized coordinates to pixel coordinates
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)
            
            w = x2 - x1
            h = y2 - y1
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            w = min(w, width - x1)
            h = min(h, height - y1)
            
            if class_id < len(labels):
                draw_detection(frame, class_id, confidence, x1, y1, w, h)
                detection_count += 1
    
    return frame, detection_count

def draw_title_bar(frame):
    """Draw instruction bar at bottom of frame"""
    height, width = frame.shape[:2]
    bar_height = 35
    bar_y = height - bar_height
    
    overlay = frame.copy()
    cv.rectangle(overlay, (0, bar_y), (width, height), (20, 20, 20), cv.FILLED)
    cv.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    cv.line(frame, (0, bar_y), (width, bar_y), (100, 100, 100), 1)
    
    title = "Press 'Q' to quit | '+/-' for confidence"
    text_size = cv.getTextSize(title, font_face, 0.4, 1)[0]
    text_x = (width - text_size[0]) // 2
    text_y = bar_y + 20
    
    cv.putText(frame, title, (text_x, text_y), font_face, 0.4, (200, 200, 200), 1, cv.LINE_AA)

def draw_performance_info(frame, fps, detection_count, inference_time):
    """Draw performance metrics in top-left corner"""
    info_lines = [f"{detection_count} Objects", f"{inference_time:.1f}ms"]
    
    padding = 8
    line_height = 18
    info_height = len(info_lines) * line_height + padding * 2
    info_width = 160
    
    overlay = frame.copy()
    cv.rectangle(overlay, (10, 10), (10 + info_width, 10 + info_height), (0, 0, 0), cv.FILLED)
    cv.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv.rectangle(frame, (10, 10), (10 + info_width, 10 + info_height), (64, 64, 64), 1)
    
    for i, line in enumerate(info_lines):
        y_pos = 25 + i * line_height
        cv.putText(frame, line, (15, y_pos), font_face, 0.45, (0, 255, 150), 1, cv.LINE_AA)

def main():
    """Main loop for real-time object detection"""
    print("Starting real-time object detection...")
    print("Press 'q' or ESC to quit")
    print("Press 's' to save current frame")
    print("Press '+' to increase confidence threshold")
    print("Press '-' to decrease confidence threshold")
    
    frame_count = 0
    fps = 0
    fps_update_interval = 30
    
    global confidence_threshold
    
    while True:
        start_time = cv.getTickCount()
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        frame = cv.flip(frame, 1)
        
        # Object detection inference
        detection_start = cv.getTickCount()
        detections = detect_objects(frame, net)
        detection_end = cv.getTickCount()
        
        # Calculate inference time
        inference_time = (detection_end - detection_start) / cv.getTickFrequency() * 1000
        
        frame, detection_count = process_detections(frame, detections)
        
        # Calculate FPS
        frame_count += 1
        if frame_count % fps_update_interval == 0:
            end_time = cv.getTickCount()
            fps = fps_update_interval / ((end_time - start_time) / cv.getTickFrequency())
        
        draw_performance_info(frame, fps, detection_count, inference_time)
        draw_title_bar(frame)
        
        cv.imshow(window_name, frame)
        
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            filename = f"detection_frame_{frame_count}.jpg"
            cv.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
        elif key == ord('+') or key == ord('='):
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
            print(f"Confidence threshold: {confidence_threshold:.2f}")
        elif key == ord('-'):
            confidence_threshold = max(0.1, confidence_threshold - 0.05)
            print(f"Confidence threshold: {confidence_threshold:.2f}")

    cap.release()
    cv.destroyAllWindows()
    print("Camera released and windows closed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        cv.destroyAllWindows()
