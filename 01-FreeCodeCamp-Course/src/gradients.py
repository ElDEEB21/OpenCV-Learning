import cv2 as cv
import numpy as np

img = cv.imread('../Photos/park.jpg')
cv.imshow('Park', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

## Gradients is the change in intensity or color in an image and can be detected using various methods, including the Sobel operator and the Laplacian.

## Laplacian (Explination: Laplacian is a second derivative filter that highlights regions of rapid intensity change)
laplacian = cv.Laplacian(gray, cv.CV_64F)
## Parameter cv.CV_64F is used to specify the depth of the output image
laplacian = np.uint8(np.absolute(laplacian))  # Convert to uint8 for display
cv.imshow('Laplacian', laplacian)
## Used to detect edges and noise in the image

## Sobel operator (Explination: Sobel operator is a first derivative filter that detects edges by calculating the gradient of the image intensity)
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
sobel = cv.bitwise_or(sobelx, sobely)  # Combine the two gradients
cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Sobel', sobel)
## Used to detect edges in the image by calculating the gradient in the x and y directions

canny = cv.Canny(gray, 150, 175)  # Canny edge detection
cv.imshow('Canny', canny)

## Differences between the methods:
# - Laplacian is a second derivative filter that highlights regions of rapid intensity change, making it useful for detecting edges and noise.
# - Sobel operator is a first derivative filter that detects edges by calculating the gradient of the image intensity, providing more control over edge detection.
# - Canny edge detection is a multi-stage algorithm that provides better edge detection by reducing noise and detecting edges at multiple thresholds.

## Finally, the best method to use depends on the specific requirements of the application, such as the level of noise in the image and the desired edge detection sensitivity.

cv.waitKey(0)
