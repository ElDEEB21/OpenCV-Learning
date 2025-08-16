import cv2 as cv
import mediapipe as mp
import time

class handDetection:
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackingConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionConfidence,
            min_tracking_confidence=self.trackingConfidence
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        
        return frame
    
    def findPosition(self, frame, handNo=0, draw=True):
        lmList = []
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED, 8)
        return lmList

def main():
    pTime, cTime = 0, 0
    
    cap = cv.VideoCapture(1)
    detector = handDetection()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)
        if len(lmList) != 0:
            print(lmList[0])  # Optional: print landmark positions

        frame = cv.flip(frame, 1)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv.putText(frame, f'FPS: {int(fps)}', (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        cv.imshow("Image", frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()