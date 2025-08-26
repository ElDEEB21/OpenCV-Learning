import cv2 as cv
import mediapipe as mp
import time

class faceMeshDetection:
    def __init__(self, staticMode=False, maxFaces=2, refine_landmarks=False, 
                 minDetectionCon=0.5, minTrackingCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refine_landmarks = refine_landmarks
        self.minDetectionCon = minDetectionCon
        self.minTrackingCon = minTrackingCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            # If True, treats input as static images (good for photos).
            # If False, treats input as a video stream and uses tracking (faster for videos).
            static_image_mode=self.staticMode,
            
            # Maximum number of faces to detect.
            max_num_faces=self.maxFaces,
            
            # If True, refines landmarks around the eyes and lips for higher quality.
            refine_landmarks=self.refine_landmarks,
            
            # Minimum confidence threshold for face detection [0.0 - 1.0].
            min_detection_confidence=self.minDetectionCon,
            
            # Minimum confidence threshold for landmark tracking [0.0 - 1.0].
            min_tracking_confidence=self.minTrackingCon
        )
        
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, frame, draw=True):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        
        faces = []
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    # Draw face mesh contours
                    self.mpDraw.draw_landmarks(frame, face_landmarks,
                                             self.mpFaceMesh.FACEMESH_CONTOURS, 
                                             self.drawSpec, self.drawSpec)
                
                # Store face landmark data
                face = []
                for id, lm in enumerate(face_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    face.append([cx, cy])
                faces.append(face)
        
        return frame, faces
    
    def findPosition(self, frame, faceNo=0, draw=True):
        lmList = []
        
        if self.results.multi_face_landmarks:
            if len(self.results.multi_face_landmarks) > faceNo:
                myFace = self.results.multi_face_landmarks[faceNo]
                for id, lm in enumerate(myFace.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        # Draw every 10th landmark point for visibility
                        if id % 10 == 0:
                            cv.circle(frame, (cx, cy), 3, (0, 255, 0), cv.FILLED)
        
        return lmList

def main():
    pTime, cTime = 0, 0
    
    cap = cv.VideoCapture(0)
    detector = faceMeshDetection(maxFaces=2)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame, faces = detector.findFaceMesh(frame)
        lmList = detector.findPosition(frame, draw=False)
        
        if len(faces) != 0:
            print(f"Found {len(faces)} face(s)")
        
        if len(lmList) != 0:
            print(f"Face has {len(lmList)} landmarks")

        frame = cv.flip(frame, 1)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv.putText(frame, f'FPS: {int(fps)}', (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv.imshow("Face Mesh", frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
