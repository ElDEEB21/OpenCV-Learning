import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('../models/haar_cascade.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('../models/face_trained.yml')

def predict_face(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face)
        print(f'Label: {people[label]}, Confidence: {confidence}')
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv.putText(img, people[label], (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return img

# img = cv.imread(r'E:\Abdo ElDeeb\Courses\Artificial Intelligence\Computer Vision\OpenCV-Learning\Faces\val\elton_john\1.jpg')
# For testing, you can use any image from the Photos folder
img = cv.imread('../Photos/group 1.jpg')
predicted_img = predict_face(img)
cv.imshow('Predicted Face', predicted_img)

cv.waitKey(0)