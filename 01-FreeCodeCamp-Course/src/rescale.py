import cv2 as cv

def rescale_frame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# img = cv.imread('../Photos/cat_large.jpg')
# resized_image = rescale_frame(img, scale=0.5)
# cv.imshow('Cat', resized_image)
# cv.waitKey(0)

# capture = cv.VideoCapture('../Videos/dog.mp4')
# while True:
#     isTrue, frame = capture.read()
    
#     if not isTrue:
#         break
    
#     resized_frame = rescale_frame(frame, scale=0.2)
#     cv.imshow('Video', resized_frame)
    
#     if cv.waitKey(20) & 0xFF == ord('d'):
#         break
# capture.release()
# cv.destroyAllWindows()

def changeResolution(width, height):
    capture.set(3, width)
    capture.set(4, height)

capture = cv.VideoCapture(0)
# changeResolution(640, 480)  

# while True:
#     isTrue, frame = capture.read()
    
#     if not isTrue:
#         break
    
#     cv.imshow('Video', frame)
    
#     if cv.waitKey(20) & 0xFF == ord('d'):
#         break
# capture.release()
# cv.destroyAllWindows()