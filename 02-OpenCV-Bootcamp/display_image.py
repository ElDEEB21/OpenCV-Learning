import cv2
import matplotlib.pyplot as plt

cb_img = cv2.imread("02-OpenCV-Bootcamp/Photos/checkerboard_color.png")
coke_img = cv2.imread("02-OpenCV-Bootcamp/Photos/coca-cola-logo.png")

# Use matplotlib imshow()
plt.imshow(cb_img)
plt.title("matplotlib imshow")
plt.show()

# Use OpenCV imshow(), display for 8 sec
cv2.imshow("w1", cb_img)
cv2.waitKey(8000)
cv2.destroyAllWindows()

# Use OpenCV imshow(), display for 8 sec
cv2.imshow("w2", coke_img)
cv2.waitKey(8000)
cv2.destroyAllWindows()

# Use OpenCV imshow(), display until any key is pressed
cv2.imshow("w3", cb_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

Alive = True
while Alive:
    # Use OpenCV imshow(), display until 'q' key is pressed
    cv2.imshow("w4", coke_img)
    keypress = cv2.waitKey(1)
    if keypress == ord('q'):
        Alive = False
cv2.destroyAllWindows()

stop = 1