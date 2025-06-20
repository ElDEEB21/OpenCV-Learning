import cv2 as cv
import sys

s = 0
if len(sys.argv) > 1:
    s = int(sys.argv[1])
# This will open the default camera (0) or the camera specified by the argument
source = cv.VideoCapture(s)
win_name = 'Camera'
cv.namedWindow(win_name, cv.WINDOW_NORMAL)
cv.resizeWindow(win_name, 640, 480)

while True:
    ret, frame = source.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv.imshow(win_name, frame)

    key = cv.waitKey(1)
    if key == 27:  # ESC key
        break
source.release()
# This will release the camera resource
cv.destroyAllWindows()
# This will close all OpenCV windows
