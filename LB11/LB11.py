from __future__ import division
from __future__ import print_function

import cv2
import sys

# from matplotlib import pyplot as plt

import numpy as np
import argparse

'''
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
'''

parser = argparse.ArgumentParser(description='Code for Histogram Calculation tutorial.')
parser.add_argument('--input', help='Path to input image.', default='D:\\dark2.png')
args = parser.parse_args()

src = cv2.imread(cv2.samples.findFile(args.input))

if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
src_bgr_planes = cv2.split(src)

histSize = 256
histRange = (0, 256)  # the upper boundary is exclusive
accumulate = False

b_hist = cv2.calcHist(src_bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
g_hist = cv2.calcHist(src_bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
r_hist = cv2.calcHist(src_bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)

hist_w = 512
hist_h = 400

bin_w = int(round(hist_w / histSize))

histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

for i in range(1, histSize):
    cv2.line(histImage, (bin_w * (i - 1), hist_h - int(b_hist[i - 1])),
             (bin_w * i, hist_h - int(b_hist[i])),
             (255, 0, 0), thickness=2)

    cv2.line(histImage, (bin_w * (i - 1), hist_h - int(g_hist[i - 1])),
             (bin_w * i, hist_h - int(g_hist[i])),
             (0, 255, 0), thickness=2)

    cv2.line(histImage, (bin_w * (i - 1), hist_h - int(r_hist[i - 1])),
             (bin_w * i, hist_h - int(r_hist[i])),
             (0, 0, 255), thickness=2)

cv2.imshow('Source image', src)

img_yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)

# equalize the histogram of the Y channel
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

# convert the YUV image back to RGB
img_yuv_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# convert from RGB to YCrCb
img_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)

# equalize the histogram of the Y channel
img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])

# convert back to RGB color-space from YCrCb
img_ycrcb_output = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)

cv2.imshow('RGB Hist', histImage)
cv2.imshow('YUV Hist', )
cv2.imshow('YCrCb hist', )

cv2.imshow('Equalized image YUV', img_yuv_output)
cv2.imshow('Equalized image YCrCb',img_ycrcb_output)

cv2.waitKey(0)
