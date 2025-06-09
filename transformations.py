import cv2 as cv
import numpy as np

img = cv.imread('Photos/park.jpg')
cv.imshow('Park', img)

## Translation (Explination: Moving the image along x and y axes)
def translate(img, x, y):
    transMat = np.float32([[1, 0, x],[0, 1, y]])
    ## transMat is a transformation matrix that shifts the image by (x, y)
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)
    ## warpAffine applies the transformation matrix to the image
## -x and -y values will move the image left and up respectively
## +x and +y values will move the image right and down respectively
translated = translate(img, -100, 100)
# cv.imshow('Translated', translated)

## Rotation (Explination: Rotating the image around its center)
def rotate(img, angle, rotPoint=None):
    (h, w) = img.shape[:2]
    
    if rotPoint is None:
        rotPoint = (w // 2, h // 2)
        ## If no rotation point is provided, use the center of the image
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    ## getRotationMatrix2D creates a rotation matrix for the given angle and scale
    dimensions = (w, h)
    return cv.warpAffine(img, rotMat, dimensions)
    ## warpAffine applies the rotation matrix to the image

rotated = rotate(img, -90)
# cv.imshow('Rotated', rotated)

rotated_rotated = rotate(rotated, -45, (100, 100))
# cv.imshow('Rotated at Point', rotated_rotated)

## Resizing (Explination: Changing the size of the image)
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_LINEAR)
# cv.imshow('Resized', resized)

## Flipping (Explination: Flipping the image horizontally or vertically)
flipped = cv.flip(img, 1)  # 0 for vertical flip, 1 for horizontal flip, -1 for both
# cv.imshow('Flipped', flipped)

## Cropping (Explination: Extracting a specific region from the image)
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)