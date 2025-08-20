import cv2 as cv
import mediapipe as mp
import time

class poseDetection:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionConf=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionConf = detectionConf
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            # If True, treats input as static images (good for photos).
            # If False, treats input as a video stream and uses tracking (faster for videos).
            static_image_mode=self.mode,
            
            # Complexity of the pose landmark model: 0 = light, 1 = medium, 2 = heavy (more accurate but slower).
            model_complexity=1,

            # If True, smooths landmark points across frames to reduce jitter.
            smooth_landmarks=self.smooth,

            # If True, enables segmentation mask (separates person from background).
            enable_segmentation=False,

            # If True, smooths the segmentation mask for a cleaner output (only used if enable_segmentation=True).
            smooth_segmentation=True,

            # Minimum confidence threshold for person detection [0.0 - 1.0].
            min_detection_confidence=self.detectionConf,

            # Minimum confidence threshold for landmark tracking [0.0 - 1.0].
            min_tracking_confidence=self.trackCon
        )

        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, frame, draw=True):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, 
                                         self.mpPose.POSE_CONNECTIONS)
        
        return frame
    
    def findPosition(self, frame, draw=True):
        lmList = []

        if self.results.pose_landmarks:
            h, w, _ = frame.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(frame, (cx, cy), 5, (255, 0, 0), cv.FILLED)
        return lmList

def main():
    pTime, cTime = 0, 0
    
    cap = cv.VideoCapture(1)
    detector = poseDetection()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = detector.findPose(frame)
        lmList = detector.findPosition(frame, False)
        if len(lmList) != 0:
            print(lmList[0]) 

        frame = cv.flip(frame, 1)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv.putText(frame, f'FPS: {int(fps)}', (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv.imshow("Pose Estimation", frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
