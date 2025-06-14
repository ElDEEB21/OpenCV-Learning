import os
import numpy as np
import cv2 as cv

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = 'faces/train'

features = [] # List to hold feature vectors
labels = [] # List to hold labels

def create_train():
    for person in people:
        img_dir = os.path.join(DIR, person)
        label = people.index(person)
        for img in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img)
            image = cv.imread(img_path)
            if image is not None:
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                face_cascade = cv.CascadeClassifier('../models/haar_cascade.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    features.append(face)
                    labels.append(label)

create_train()

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(np.array(features, dtype='object'), np.array(labels))

face_recognizer.save('../models/face_trained.yml')

# Convert features to numpy array with object dtype to handle different sized faces
features_array = np.array(features, dtype=object)
labels_array = np.array(labels)

np.save('../models/features.npy', features_array)
np.save('../models/labels.npy', labels_array)

