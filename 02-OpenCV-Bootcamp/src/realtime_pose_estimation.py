import cv2 as cv
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level from src/

protoFile = os.path.join(script_dir, "Models", "pose_deploy_linevec_faster_4_stages.prototxt")
weightsFile = os.path.join(script_dir, "Models", "pose_iter_160000.caffemodel")

try:
    net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

camera_source = 0
if len(sys.argv) > 1:
    camera_source = int(sys.argv[1])

cap = cv.VideoCapture(camera_source)
if not cap.isOpened():
    print(f"Error: Could not open camera {camera_source}")
    sys.exit(1)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv.CAP_PROP_FPS, 15)
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))

window_name = 'Real-time Pose Estimation'
cv.namedWindow(window_name, cv.WINDOW_NORMAL)

nPoints = 15
netInputSize = (192, 192)
confidence_threshold = 0.2

POSE_PAIRS = [
    [0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14],
    [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13],
]

keypoint_colors = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
]

font_face = cv.FONT_HERSHEY_SIMPLEX
font_scale = 0.4
font_thickness = 1

def detect_pose(frame, net):
    inpBlob = cv.dnn.blobFromImage(frame, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
    return output

def extract_keypoints(output, frame_width, frame_height):
    scaleX = frame_width / output.shape[3]
    scaleY = frame_height / output.shape[2]
    
    points = []
    
    for i in range(nPoints):
        probMap = output[0, i, :, :]
        minVal, confidence, minLoc, point = cv.minMaxLoc(probMap)
        
        x = int(point[0] * scaleX)
        y = int(point[1] * scaleY)
        
        if confidence > confidence_threshold:
            points.append((x, y))
        else:
            points.append(None)
    
    return points

def draw_keypoints(frame, points):
    for i, point in enumerate(points):
        if point is not None:
            color = keypoint_colors[i % len(keypoint_colors)]
            cv.circle(frame, point, 3, color, thickness=-1)

def draw_skeleton(frame, points):
    for pair in POSE_PAIRS:
        partA, partB = pair
        
        if points[partA] is not None and points[partB] is not None:
            color = keypoint_colors[partA % len(keypoint_colors)]
            cv.line(frame, points[partA], points[partB], color, 1)

def draw_performance_info(frame, fps, keypoint_count, inference_time):
    if frame.shape[0] > 200:  # Only draw if frame is large enough
        info = f"FPS:{fps:.0f} Pts:{keypoint_count} Ms:{inference_time:.0f}"
        cv.putText(frame, info, (10, 25), font_face, 0.5, (0, 255, 150), 1)

def draw_title_bar(frame):
    height, width = frame.shape[:2]
    if height > 50:  # Only draw if frame is large enough
        title = "Q:quit +/-:conf K:keypts L:skel"
        cv.putText(frame, title, (10, height - 10), font_face, 0.4, (200, 200, 200), 1)

def main():
    global confidence_threshold
    
    frame_count = 0
    fps = 0
    fps_update_interval = 20
    show_keypoints = True
    show_skeleton = True
    
    process_every_n_frames = 4
    last_points = None
    
    # Pre-calculate timing variables
    start_time = cv.getTickCount()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        
        if frame_count % process_every_n_frames == 0:
            detection_start = cv.getTickCount()
            output = detect_pose(frame, net)
            detection_end = cv.getTickCount()
            
            inference_time = (detection_end - detection_start) / cv.getTickFrequency() * 1000
            last_points = extract_keypoints(output, frame_width, frame_height)
        else:
            inference_time = 0
        
        points = last_points if last_points is not None else []
        keypoint_count = sum(1 for point in points if point is not None) if points else 0
        
        if show_skeleton and points:
            draw_skeleton(frame, points)
        
        if show_keypoints and points:
            draw_keypoints(frame, points)
        
        frame_count += 1
        if frame_count % fps_update_interval == 0:
            end_time = cv.getTickCount()
            fps = fps_update_interval / ((end_time - start_time) / cv.getTickFrequency())
            start_time = end_time
        
        draw_performance_info(frame, fps, keypoint_count, inference_time)
        draw_title_bar(frame)
        
        cv.imshow(window_name, frame)
        
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            filename = f"pose_frame_{frame_count}.jpg"
            cv.imwrite(filename, frame)
        elif key == ord('+') or key == ord('='):
            confidence_threshold = min(0.9, confidence_threshold + 0.05)
        elif key == ord('-'):
            confidence_threshold = max(0.05, confidence_threshold - 0.05)
        elif key == ord('k'):
            show_keypoints = not show_keypoints
        elif key == ord('l'):
            show_skeleton = not show_skeleton
        elif key == ord('f'):
            process_every_n_frames = 6 if process_every_n_frames == 4 else 4
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        cv.destroyAllWindows()
