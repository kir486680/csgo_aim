import cv2
import mss
import numpy as np
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")
sct = mss.mss()
img_counter = 0
monitor = {"top": 40, "left": 0, "width": 1920, "height": 1080}

while True:
    frame = sct.grab(monitor)
    frame = np.array(frame)
  
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    # SPACE pressed
    img_name = "dataset/theopencv_frame_t{}.jpg".format(img_counter)
    cv2.imwrite(img_name, frame)
    print("{} written!".format(img_name))
    img_counter += 1

cam.release()

cv2.destroyAllWindows()
