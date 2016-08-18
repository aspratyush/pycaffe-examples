#/bin/python

#Load a basic prototxt file and perform simple computations

import numpy as np
import caffe
import cv2
import matplotlib.pyplot as plt

#Load the net
net = caffe.Net("../models/conv.prototxt", caffe.TEST)

#Load the image and typecast to np.array
img = cv2.imread("../images/cat_gray.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.array(img)

#Reshape to 4D-Tensor (1x1xMxN)
img_input = img[np.newaxis, np.newaxis, :, :]

#Reshape I/P of net to accept this data and load data
net.blobs['data'].reshape(*img_input.shape)
net.blobs['data'].data[...] = img_input

#net forward once
net.forward()
