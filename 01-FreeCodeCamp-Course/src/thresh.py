import cv2 as cv

img = cv.imread('../Photos/cats.jpg')
cv.imshow('Cats', img)

## Thresholding is a way to create a binary image from a grayscale image

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

## Simple Thresholding (Explination: If pixel value is greater than a threshold, set it to max value, else set it to 0)
_, thresh1 = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
## Parameters: (source image, threshold value, max value, thresholding type)
## Output: thresh1 is a binary image where pixel values are either 0 or 255
## Used when the image has uniform lighting conditions
cv.imshow('Simple Thresholding', thresh1)

## Inverse Thresholding (Explination: If pixel value is greater than a threshold, set it to 0, else set it to max value)
_, thresh2 = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow('Inverse Thresholding', thresh2)

## Adaptive Thresholding (Explination: Threshold value is calculated for smaller regions, allowing for different lighting conditions)
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
## Parameters: (source image, max value, adaptive method, thresholding type, block size, constant subtracted from the mean)
## Output: adaptive_thresh is a binary image where pixel values are either 0 or 255 based on local mean
## Used when the image has varying lighting conditions
cv.imshow('Adaptive Thresholding', adaptive_thresh)

cv.waitKey(0)