import cv2 as cv

def display_image():
    img = cv.imread('../Photos/cat_large.jpg')
    cv.imshow('Cat', img)
    cv.waitKey(0)

def play_video():
    capture = cv.VideoCapture('../Videos/dog.mp4')
    while True:
        isTrue, frame = capture.read()
        
        if not isTrue:
            break
            
        cv.imshow('Video', frame)
            
        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    capture.release()
    cv.destroyAllWindows()
