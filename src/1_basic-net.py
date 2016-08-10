#/bin/python

# Load a basic prototxt file and perform simple computations

import numpy as np
import caffe
import cv2
import matplotlib.pyplot as plt

#Load the net
net = caffe.Net("../models/conv.prototxt", caffe.TEST)

#Load the image and typecast to np.array
img = np.array(cv2.imread("../images/cat_gray.jpg"))

