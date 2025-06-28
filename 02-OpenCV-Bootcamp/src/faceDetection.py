import cv2 as cv
import sys
import os

s = 0
if len(sys.argv) > 1:
    s = int(sys.argv[1])

source = cv.VideoCapture(s)
winName = 'Face Detection'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level from src/
prototxt_path = os.path.join(script_dir, 'Models/deploy.prototxt')
# This file is a pre-trained model for face detection
model_path = os.path.join(script_dir, 'Models/res10_300x300_ssd_iter_140000_fp16.caffemodel')
# This file is a pre-trained model for face detection using Caffe framework

# The Difference between the two files is that the prototxt file contains the model architecture,
# while the caffemodel file contains the pre-trained weights.

net = cv.dnn.readNetFromCaffe(prototxt_path, model_path)
# This function reads the model architecture from the prototxt file and the pre-trained weights from the caffemodel file.

# Model Parameters
inWidth = 300
inHeight = 300
mean = [104, 117, 123] # Mean values for each channel (BGR) to be subtracted from the input image
# These values are used to normalize the input image before feeding it to the neural network.
confThreshold = 0.7 # Confidence threshold for filtering weak detections

while cv.waitKey(1) != 27:
    hasFrame, frame = source.read()
    if not hasFrame:
        break
    
    frame = cv.flip(frame, 1) # Flip the frame horizontally
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    
    # Create a 4D blob from the frame
    # The blob is a 4D array with shape (1, 3, inHeight, inWidth), used as input to the neural network.
    blob = cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), mean, swapRB=False, crop=False)
    # The parameters are:
    # - firsst argument is the input image
    # - second argument is the scale factor (1.0 means no scaling)
    # - third argument is the size of the input image (inWidth, inHeight)
    # - fourth argument is the mean subtraction values (mean values for each channel)
    # - swapRB is set to False because the model expects the input in BGR format
    # - crop is set to False because we want to resize the image without cropping it
    
    # Run Model
    net.setInput(blob)
    detections = net.forward()
    # Detections is a 4D array with shape (1, 1, N, 7), where N is the number of detected faces.
    # Each detection contains the following information:
    # - detections[0, 0, i, 0] - detection confidence
    # - detections[0, 0, i, 1] - class label
    # - detections[0, 0, i, 2] - confidence score
    # - detections[0, 0, i, 3] - x-coordinate
    # - detections[0, 0, i, 4] - y-coordinate
    # - detections[0, 0, i, 5] - width of the bounding box
    # - detections[0, 0, i, 6] - height of the bounding box
    # The first two dimensions are batch size and number of classes, respectively.
    # The third dimension contains the detections, and the fourth dimension contains the detection information.
    
    for i in range(detections.shape[2]):
        # Loop through each detection in the third dimension of the detections array
        confidence = detections[0, 0, i, 2]
        # Extract the confidence score for the current detection (index 2 contains confidence)
        if confidence > confThreshold:
            # Only process detections with confidence above the threshold
            xLeftBottom = int(detections[0, 0, i, 3] * frameWidth)
            # Calculate the x-coordinate of the left edge of the bounding box (normalized to frame width)
            yLeftBottom = int(detections[0, 0, i, 4] * frameHeight)
            # Calculate the y-coordinate of the bottom edge of the bounding box (normalized to frame height)
            xRightTop = int(detections[0, 0, i, 5] * frameWidth)
            # Calculate the x-coordinate of the right edge of the bounding box (normalized to frame width)
            yRightTop = int(detections[0, 0, i, 6] * frameHeight)
            # Calculate the y-coordinate of the top edge of the bounding box (normalized to frame height)
            
            # Draw main bounding box with rounded corners effect
            cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 255), thickness=3)
            label = "Face: %.1f%%" % (confidence * 100)
            
            # Position label above the bounding box with padding
            label_y, label_x = yLeftBottom - 10, xLeftBottom

            # Display the confidence label text in bright cyan
            cv.putText(frame, label, (label_x, label_y),
                       cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv.LINE_AA)
            
    t, _ = net.getPerfProfile() # Get the inference time
    # inference time is the time taken by the neural network to process the input image
    label = 'Inference: %.1f ms' % (t * 1000.0 / cv.getTickFrequency())
    # cv.getTickFrequency() returns the number of ticks per second for the system
    
    # Draw performance info with better styling
    label_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv.rectangle(frame, (5, 5), (label_size[0] + 15, 30), (0, 0, 0), cv.FILLED)
    cv.putText(frame, label, (10, 22), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)
    cv.imshow(winName, frame)