import cv2 as cv
import numpy as np

img = cv.imread('../Photos/cats.jpg')
# cv.imshow('Cat', img)

blank = np.zeros(img.shape, dtype='uint8')  # Create a blank image with the same shape as the original image
cv.imshow('Blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
# cv.imshow('Blurred', blur)

# canny = cv.Canny(blur, 125, 175)
# cv.imshow('Canny Edges', canny)

## Explanation: This applies a binary threshold to the grayscale image.
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
## Parameters: source image, threshold value, max value to use with the THRESH_BINARY type, thresholding type
## ret is the threshold value used, which is returned by the function
## thresh is the output binary image where pixels below the threshold are set to 0 and those above are set to 255
cv.imshow('Thresholded', thresh)

## Explanation: This finds the contours in the image. Contours are curves joining all the continuous points along a boundary that have the same color or intensity.
contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
## Parameters: source image, contour retrieval mode, contour approximation method
## Explanation of parameters:
## - cv.RETR_LIST: retrieves all contours without establishing any hierarchical relationships.
## - cv.CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments and leaves only their end points.
## contours is a list of all the contours found in the image
## hierarchy is a list that contains information about the image topology

print(len(contours))  # Prints the number of contours found

## Explanation: This draws the contours on the blank image.
cv.drawContours(blank, contours, -1, (0, 255, 0), thickness=1)
## Parameters: image to draw on, contours to draw, index of the contour (-1 means all contours), color of the contour, thickness of the contour
cv.imshow('Contours', blank)

cv.waitKey(0)