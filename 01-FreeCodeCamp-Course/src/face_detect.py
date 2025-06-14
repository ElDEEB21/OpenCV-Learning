import cv2 as cv

img = cv.imread('../Photos/group 1.jpg')
cv.imshow('Photo', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

## Explination:
# The Haar Cascade classifier is a machine learning object detection method used to identify objects for which it has been trained.
haar_cascade = cv.CascadeClassifier('../models/haar_cascade.xml')
### The CascadeClassifier class is used to load the pre-trained Haar Cascade model for face detection.
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
## Parameters:
# scaleFactor: Specifies how much the image size is reduced at each image scale.
# minNeighbors: Specifies how many neighbors each candidate rectangle should have to retain it.

print(f'Number of faces detected: {len(faces_rect)}')

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    # Draw a rectangle around the detected face
cv.imshow('Detected Faces', img)

cv.waitKey(0)

## haar_face.xml
# is a pre-trained model file for face detection using Haar Cascade.
# It contains the necessary data for the classifier to detect faces in images.
# It describes the features of a face and how to recognize them based on the training data it has been provided.