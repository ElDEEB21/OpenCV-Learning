import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')
# cv.imshow('Blank', blank)

# blank[:] = 0, 255, 0  # Fill the image with green color
# cv.imshow('Green', blank)

# blank[:] = 255, 0, 0  # Change the color to blue
# cv.imshow('Blue', blank)

# blank[:] = 0, 0, 255  # Change the color to red
# cv.imshow('Red', blank)

# blank[200:300, 300:400] = 255, 255, 0  # Draw a light blue square
# cv.imshow('Light blue Square', blank)


# cv.rectangle(blank, (0, 0), (blank.shape[1] // 2, blank.shape[0] // 2), (255, 0, 0), thickness=cv.FILLED)  # Draw a filled blue rectangle
# image, top-left corner, bottom-right corner, color, thickness
# cv.imshow('Rectangle', blank)

# cv.circle(blank, (blank.shape[1] // 2, blank.shape[0] // 2), 40, (0, 255, 0), thickness=cv.FILLED)  # Draw a filled green circle
# image, center, radius, color, thickness
# cv.imshow('Circle', blank)

# cv.line(blank, (0, 0), (blank.shape[1], blank.shape[0]), (0, 0, 255), thickness=3)  # Draw a red diagonal line
# image, start point, end point, color, thickness
# cv.imshow('Line', blank)

# cv.putText(blank, 'OpenCV', (225, 250), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), thickness=2)  # Add text to the image
# image, text, position, font, font scale, color, thickness
# cv.imshow('Text', blank)

cv.waitKey(0)