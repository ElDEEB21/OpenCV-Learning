import cv2 as cv
import numpy as np
import sys

PREVIEW = 0
BLUR = 1
FEATURES = 2
CANNY = 3

feature_params = {
    'maxCorners': 500,
    'qualityLevel': 0.2,
    'minDistance': 15,
    'blockSize': 9
}

s = 0
if len(sys.argv) > 1:
    s = int(sys.argv[1])

image_filter = PREVIEW
alive = True

win_name = 'Camera Filters'
cv.namedWindow(win_name, cv.WINDOW_NORMAL)
result = None

source = cv.VideoCapture(s)
if not source.isOpened():
    print("Error: Could not open video source.")
    sys.exit()

while alive:
    has_frame, frame = source.read()
    if not has_frame:
        break

    frame = cv.flip(frame, 1)

    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        result = cv.Canny(frame, 80, 150)
    elif image_filter == BLUR:
        result = cv.blur(frame, (13, 13))
    elif image_filter == FEATURES:
        result = frame
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(frame_gray, **feature_params)
        # goodFeaturesToTrack used to find corners in the image
        # Parameters:
        # maxCorners: maximum number of corners to return
        # qualityLevel: parameter characterizing the minimal accepted quality of image corners
        # minDistance: minimum possible Euclidean distance between the returned corners
        # blockSize: size of an average block for computing a derivative covariation matrix over each pixel neighborhood
        # Returns:
        # corners: a list of detected corners in the image
        if corners is not None:
            for x, y in np.float32(corners).reshape(-1, 2):
                cv.circle(result, (int(x), int(y)), 10, (0, 255, 0), 1)
            # The loop iterates over the detected corners and draws a circle around each corner
            # reshape is used to convert the corners to a 2D array of coordinates

    cv.imshow(win_name, result)

    key = cv.waitKey(1)
    if key == ord("Q") or key == ord("q") or key == 27:
        alive = False
    elif key == ord("C") or key == ord("c"):
        image_filter = CANNY
    elif key == ord("B") or key == ord("b"):
        image_filter = BLUR
    elif key == ord("F") or key == ord("f"):
        image_filter = FEATURES
    elif key == ord("P") or key == ord("p"):
        image_filter = PREVIEW

source.release()
cv.destroyAllWindows()
