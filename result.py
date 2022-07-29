# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 01:48:40 2022

@author: swaroop hn and palash sarate
"""
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
cv2.equalizeHist

# Read the image
img = cv2.imread("C:\Download/71.tiff", 0)

# Thresholding the image
(thresh, img_bin) = cv2.threshold(img, 128, 255,cv2.THRESH_BINARY|     cv2.THRESH_OTSU)

img_bin = 255-img_bin 
cv2.imwrite("Image_bin.jpg",img_bin)

bw = cv2.adaptiveThreshold(img_bin, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                            cv2.THRESH_BINARY, 15, -2)
# Defining a kernel length
kernel_length = np.array(img).shape[1]//80

# A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

# A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

# A kernel of (3 X 3) ones.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Morphological operation to detect vertical lines from an image
img_temp1 = cv2.erode(bw, verticle_kernel, iterations=3)

# vertical_lines_img = cv2.morphologyEx(img_temp1, cv2.MORPH_OPEN, verticle_kernel, iterations=1)
verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=15)
# verticle_lines_img = cv2.erode(verticle_lines_img, verticle_kernel, iterations=9)
cv2.imwrite("C:\Download/v_in.tiff",verticle_lines_img)

# Morphological operation to detect horizontal lines from an image
img_temp2 = cv2.erode(bw, hori_kernel, iterations=3)

# horizontal_line_img = cv2.morphologyEx(img_temp2, cv2.MORPH_OPEN, hori_kernel, iterations=1)
horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=10)
# horizontal_lines_img = cv2.dilate(horizontal_lines_img, hori_kernel, iterations=5)
cv2.imwrite("C:\Download/h_in.tiff",horizontal_lines_img)

# fusion
I = verticle_lines_img + horizontal_lines_img 
# kernel = np.ones((1,1),np.uint8)
# erosion = cv2.erode(I,kernel,iterations = 1)
cv2.imwrite("C:\Download/free1.tiff",I)

# Bitwise-and masks together
result = 255 - cv2.bitwise_or(verticle_lines_img, horizontal_lines_img)

# Fill individual grid holes
cnts = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    # cv2.rectangle(result, (x, y), (x + w, y + h), 255, 1)
equalized = cv2.equalizeHist(result)

kernel = np.array([[-1,-1,-1], 
                   [-1, 9,-1],
                   [-1,-1,-1]])
sharpened = cv2.filter2D(equalized, -1, kernel) # applying the sharpening kernel to the input image & displaying it.
cv2.imwrite("C:\Download/relt0.tiff",sharpened)

cv2.imshow('vertical_mask', verticle_lines_img)
cv2.imshow('horizontal_mask', horizontal_lines_img)
cv2.imshow('fusion',I)
cv2.imshow('result', sharpened)
cv2.waitKey()


