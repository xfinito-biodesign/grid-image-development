# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 06:04:06 2022

@author: swaroop hn and palash sarate
"""
from skimage import data, io, segmentation, color
from skimage.filters import threshold_triangle
import matplotlib.pyplot as plt
import cv2
from skimage import data
from skimage.filters import try_all_threshold
image = cv2.imread("C:\Download/LEFT_1.tiff", 0)
# image = data.camera()
thresh = threshold_triangle(image)
binary = image > thresh
b =binary
io.imsave(r"C:\Users\swaroop hn\Desktop\7.tiff",binary)
# fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
# ax = axes.ravel()
# ax[0] = plt.subplot(1, 3, 1)
# ax[1] = plt.subplot(1, 3, 2)
# ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

# ax[0].imshow(image, cmap=plt.cm.gray)
# ax[0].set_title('Original')
# ax[0].axis('off')

# ax[1].hist(image.ravel(), bins=256)
# ax[1].set_title('Histogram')
# ax[1].axvline(thresh, color='r')

# ax[2].imshow(binary, cmap=plt.cm.gray)
# ax[2].set_title('Thresholded')
# ax[2].axis('off')

# plt.show()
