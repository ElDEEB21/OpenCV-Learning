import cv2 as cv

img = cv.imread('Photos/cat.jpg')

cv.imshow('Cat', img)

## Convert to grayscale (Explaination: This converts the image from BGR to grayscale)
gray  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
## Parameters: source image, color conversion code
# cv.imshow('Gray Cat', gray)

## Blur the image (Explaination: This applies a Gaussian blur to the image, which helps reduce noise and detail)
blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
## Parameters: source image, kernel size, border type
# cv.imshow('Blurred Cat', blur)

## Edge Cascade Detection (Explaination: This applies the Canny edge detection algorithm to find edges in the image)
canny = cv.Canny(blur, 125, 175)
## Parameters: source image, lower threshold, upper threshold (Threshold: is the value that determines how strong the edges should be)
# cv.imshow('Canny Edges', canny)

## Dilating the image (Explaination: This increases the white region in the image, which can help in highlighting the edges)
dilated = cv.dilate(canny, (7, 7), iterations=3)
## Parameters: source image, kernel size, number of iterations (Explains how many times to apply the dilation)
# cv.imshow('Dilated Edges', dilated)

## Eroding the image (Explaination: This reduces the white region in the image, which can help in removing noise)
eroded = cv.erode(dilated, (7, 7), iterations=3)
## Parameters: source image, kernel size, number of iterations
# cv.imshow('Eroded Edges', eroded)

## Resize the image (Explaination: This changes the size of the image to 500x500 pixels)
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_LINEAR)
## Parameters: source image, new size, interpolation method (interpolation method: determines how the pixels are resampled when resizing)
cv.imshow('Resized Cat', resized)

## Cropping the image (Explaination: This extracts a specific region from the image)
cropped = img[50:200, 200:400]
## Parameters: source image, [y1:y2, x1:x2] (y1:y2: vertical range, x1:x2: horizontal range)
cv.imshow('Cropped Cat', cropped)

cv.waitKey(0)