import cv2
import sys

img = cv2.imread("D:\\gop.jpg")
if img is None:
    sys.exit("Could not read the image.")
cv2.imshow("Display window", img)
key = cv2.waitKey(0)
if key == ord("c"):
    cv2.imwrite("gop.jpg", img)