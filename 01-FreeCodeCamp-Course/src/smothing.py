import cv2 as cv

img = cv.imread('../Photos/cats.jpg')
cv.imshow('Cat', img)

## Averaging (Explination: This method replaces each pixel's value with the average of its neighbors)
avg = cv.blur(img, (3, 3))
cv.imshow('Averaging', avg)
## Used to reduce image noise and detail, specifically for uniform noise

## Gaussian Blurring (Explination: This method uses a Gaussian kernel
gaussian = cv.GaussianBlur(img, (3, 3), 0)
cv.imshow('Gaussian Blurring', gaussian)
## Used to reduce image noise and detail, specifically for Gaussian noise

## Median Blurring (Explination: This method replaces each pixel's value with the median of its neighbors)
median = cv.medianBlur(img, 3)
cv.imshow('Median Blurring', median)
## Used to reduce image noise and detail, specifically for salt-and-pepper noise

## Bilateral Blurring (Explination: This method replaces each pixel's value with a weighted average of its neighbors, considering both spatial distance and intensity difference)
bilateral = cv.bilateralFilter(img, 9, 75, 75)
## Parameters Explination:
# 9: Diameter of the pixel neighborhood
# 75: Sigma in color space (larger values mean more colors will be considered similar)
# 75: Sigma in coordinate space (larger values mean pixels farther away will be considered similar)
cv.imshow('Bilateral Blurring', bilateral)
# Used to reduce image noise while preserving edges, specifically for images with varying noise levels

cv.waitKey(0)