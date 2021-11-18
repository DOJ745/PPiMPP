import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np

path = "D:\\gop.jpg"
img = cv2.imread(path)

if img is None:
    sys.exit("Could not read the image.")
cv2.imshow("Basic image", img)

img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow("Gray image", img_grey)

# define a threshold, 128 is the middle of black and white in grey scale
thresh = 128
img_binary = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)[1]

cv2.imshow('Black and White', img_binary)

key = cv2.waitKey(0)
if key == ord("n"):
    cv2.imwrite("gopNormal.jpg", img)

if key == ord("g"):
    cv2.imwrite("gopGrey.jpg", img_grey)

if key == ord("b"):
    cv2.imwrite("gopBlackWhite.jpg", img_binary)
