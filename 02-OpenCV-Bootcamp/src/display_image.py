import cv2
import matplotlib.pyplot as plt
import os

# Get the path relative to the project root
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level from src/
cb_img = cv2.imread(os.path.join(script_dir, "Photos/checkerboard_color.png"))
coke_img = cv2.imread(os.path.join(script_dir, "Photos/coca-cola-logo.png"))

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