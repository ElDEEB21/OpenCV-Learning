import cv2 as cv
import mediapipe as mp
import time

class faceDetection:
    def __init__(self, minDetectionConf=0.5):
        self.minDetectionConf = minDetectionConf

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            # Model selection: 0 for faces within 2 meters, 1 for faces within 5 meters
            model_selection=0,
            
            # Minimum confidence threshold for face detection [0.0 - 1.0]
            min_detection_confidence=self.minDetectionConf
        )
        
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, frame, draw=True):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # Get bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                
                if draw:
                    # Draw fancy rectangle with corner decorations
                    self.fancyDraw(frame, bbox, detection.score[0])
        
        return frame, bboxs
    
    def findPosition(self, frame, draw=True):
        lmList = []
        
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # Get facial keypoints if available
                if detection.location_data.relative_keypoints:
                    for i, keypoint in enumerate(detection.location_data.relative_keypoints):
                        ih, iw, ic = frame.shape
                        x, y = int(keypoint.x * iw), int(keypoint.y * ih)
                        lmList.append([i, x, y])
        
        return lmList

    
    def fancyDraw(self, frame, bbox, score, l=30, t=5, rt=1):
        """
        Draw fancy rectangle with corner decorations
        bbox: (x, y, w, h)
        l: length of corner lines
        t: thickness of corner lines
        rt: corner line thickness
        """
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        
        # Main rectangle
        cv.rectangle(frame, bbox, (255, 0, 255), rt)
        
        # Top Left Corner
        cv.line(frame, (x, y), (x + l, y), (255, 0, 255), t)
        cv.line(frame, (x, y), (x, y + l), (255, 0, 255), t)
        
        # Top Right Corner
        cv.line(frame, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv.line(frame, (x1, y), (x1, y + l), (255, 0, 255), t)
        
        # Bottom Left Corner
        cv.line(frame, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv.line(frame, (x, y1), (x, y1 - l), (255, 0, 255), t)
        
        # Bottom Right Corner
        cv.line(frame, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv.line(frame, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        
        # Confidence score - positioned to appear correctly when mirrored
        cv.putText(frame, f'{int(score * 100)}%', 
                  (x1 - 80, y - 10), cv.FONT_HERSHEY_PLAIN, 
                  2, (255, 0, 255), 2)

def main():
    pTime, cTime = 0, 0
    
    cap = cv.VideoCapture(1)
    detector = faceDetection()
    
    while True:
        success, frame = cap.read()
        frame = cv.flip(frame, 1)
        if not success:
            break
        
        frame, bboxs = detector.findFaces(frame)
        
        if len(bboxs) != 0:
            print(f"Found {len(bboxs)} face(s)")

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv.putText(frame, f'FPS: {int(fps)}', (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv.imshow("Face Detection", frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
