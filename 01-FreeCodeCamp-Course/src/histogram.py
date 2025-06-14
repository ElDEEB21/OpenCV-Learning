import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('../Photos/cats.jpg')
cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blank = np.zeros(img.shape[:2], dtype='uint8')

mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)

masked = cv.bitwise_and(gray, gray, mask=mask)
cv.imshow('Mask', masked)

## Explination: this function calculates the histogram of an image.
## The histogram is a representation of the distribution of pixel intensities in the image.
gray_hist = cv.calcHist([gray], [0], masked, [256], [0, 256])
### Parameters:
## - images: The source image(s) for which the histogram is to be calculated.
## - channels: The specific channel(s) for which the histogram is to be computed.
## - mask: An optional mask to specify a region of interest.
## Note: When making mask, the calculated histogram will only consider the pixels within the masked area.
## - histSize: The number of bins in the histogram.
## - ranges: The range of pixel values to consider.

plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('Pixel Count')
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()

## Color Histogram
## Explanation: This section calculates the histogram for each color channel (B, G, R) in the image.
colores = ('b', 'g', 'r')
for i, color in enumerate(colores):
    hist = cv.calcHist([img], [i], mask, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('Pixel Count')
plt.show()

cv.waitKey(0)