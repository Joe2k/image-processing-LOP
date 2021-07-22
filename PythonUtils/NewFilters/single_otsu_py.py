# -*- coding: utf-8 -*-
"""single_Otsu.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KQKRUa9GNmqCD_072yNP35tFRI83Hko1
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import skimage.measure
import skimage.morphology

from google.colab import files
uploaded = files.upload()

img1 = cv2.imread('2.png')
img = cv2.imread('2.png', 0)

# Applying Histogram Equalization on the original image
image_equalized = cv2.equalizeHist(img)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(image_equalized,(5,5),0)
ret3,th3 = cv2.threshold(blur,150,230,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.subplot(121),plt.imshow(img, cmap='gray')
plt.title('contrast'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(th3, cmap = 'gray')
plt.title('otsu'), plt.xticks([]), plt.yticks([])

plt.show()

def vdk_perimeter(th3):
    (w, h) = th3.shape
    data = np.zeros((w + 2, h + 2), dtype=th3.dtype)
    data[1:-1, 1:-1] = th3
    data = skimage.morphology.binary_dilation(data)
    newdata = np.copy(data)
    for i in range(1, w + 1):
        for j in range(1, h + 1):
            cond = data[i, j] == data[i, j + 1] and \
                   data[i, j] == data[i, j - 1] and \
                   data[i, j] == data[i + 1, j] and \
                   data[i, j] == data[i - 1, j]
            if cond:
                newdata[i, j] = 0

    return np.count_nonzero(newdata)

label_img = skimage.measure.label(th3)
regions = skimage.measure.regionprops(label_img)

for props in regions:
    print (props.area, vdk_perimeter(props.convex_image))