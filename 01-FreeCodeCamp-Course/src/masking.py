import cv2 as cv
import numpy as np

img = cv.imread('../Photos/cats 2.jpg')
cv.imshow('Cats', img)

blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow('Blank', blank)

# Create a circular mask at the center of the image with radius 100
mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
cv.imshow('Mask', mask)

# Apply the circular mask to the original image
masked_image = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked Image', masked_image)

# Create a rectangular mask at the center of the image (200x200 pixels)
mask2 = cv.rectangle(blank, (img.shape[1]//2 - 100, img.shape[0]//2 - 100),
                      (img.shape[1]//2 + 100, img.shape[0]//2 + 100), 255, -1)

# Apply the rectangular mask to the original image
masked_image2 = cv.bitwise_and(img, img, mask=mask2)
cv.imshow('Masked Image 2', masked_image2)

cv.waitKey(0)