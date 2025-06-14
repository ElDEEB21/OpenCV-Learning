import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('../Photos/park.jpg')
cv.imshow('Park', img)

# plt.imshow(img)
# plt.show()

## BGR is different from RGB, OpenCV uses BGR by default, but matplotlib uses RGB.

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

## BGR to HSV (Hue, Saturation, Value)
## Hue represents the color type, Saturation represents the intensity of the color, and Value represents the brightness of the color.
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.imshow('HSV', hsv)
## Used in color detection, image segmentation, and object tracking.

## BGR to LAB (Luminance, A channel, B channel)
## Luminance represents the brightness of the color, A channel represents the green-red component, and B channel represents the blue-yellow component.
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)
## Used in color correction, image enhancement, and color space conversion.

## BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# cv.imshow('RGB', rgb)
# plt.imshow(rgb)
# plt.show()

## matplot will show the image in RGB format, so it will look different from the OpenCV window.

## HSV to BGR 
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
# cv.imshow('HSV to BGR', hsv_bgr)

## LAB to BGR
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow('LAB to BGR', lab_bgr)

cv.waitKey(0)