import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import sys
import os
import math

# Add the hand tracking module path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '01-Hand-Tracking'))
import HandTrackingModule as htm

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

wCam, hCam = 640, 480

cap = cv.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize hand detector
detector = htm.handDetection(maxHands=1, detectionConfidence=0.7)

cTime, pTime = 0, 0

def draw_simple_volume_bar(frame, volPer, volBar):
    """Draw a simple volume bar"""
    # Outer rectangle
    cv.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
    
    # Volume fill
    cv.rectangle(frame, (50, int(volBar)), (85, 400), (0, 255, 0), cv.FILLED)
    
    # Volume percentage text
    cv.putText(frame, f'{int(volPer)}%', (40, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv.flip(frame, 1)
    
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip
        
        # Draw finger points
        cv.circle(frame, (x1, y1), 8, (255, 0, 255), cv.FILLED)
        cv.circle(frame, (x2, y2), 8, (255, 0, 255), cv.FILLED)

        # Draw line between fingers
        cv.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        
        # Calculate distance
        length = math.hypot(x2 - x1, y2 - y1)
        
        # Center point
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv.circle(frame, (cx, cy), 8, (255, 0, 255), cv.FILLED)

        # Volume calculations
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])

        volume.SetMasterVolumeLevel(vol, None)

        # Draw simple volume bar
        draw_simple_volume_bar(frame, volPer, volBar)
        
        # Change color when close (muted)
        if length < 50:
            cv.circle(frame, (cx, cy), 8, (0, 0, 255), cv.FILLED)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv.putText(frame, f'FPS: {int(fps)}', (40, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    
    cv.imshow("Volume Hand Control", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
