#!/bin/python

#Logistic regression using the brand-age-gender dataset

import numpy as np
import matplotlib.pyplot as plt
import caffe
import csv

# function to summarize the data read from the file"
def summarizeData(data, labels):
    print "=================================================="
    print "          Input Data statistics"
    print ""
    print "No. of examples = ", data.shape[0]
    print "Brands = ", np.unique(labels)
    - np.int( np.min(labels)) + 1
    print "Min female = ", np.int( np.min(data[:,0]))
    print "Max female = ", np.int( np.max(data[:,0]))
    print "Min Age = ", np.int(np.min(data[:,1]))
    print "Max Age = ", np.int(np.max(data[:,1]))
    print "=================================================="


# Open data in a list
dataList = []
for i in csv.reader( open('./data/example-logistic-regression.csv') ):
    dataList.append(i)

# Convert to np-array
dataArray = np.array(dataList)
#Convert to float and drop the 1st column
dataArray = dataArray[1:,:].astype(np.float32)

# separate data and labels
labels = dataArray[:,1]
data = dataArray[:,[2,3]]

#change labels to zero-index
labels -= 1

summarizeData(data, labels)

# Load the model
solver = caffe.get_solver('./models/solver_prac3.prototxt')

#convert data and labels into contiguous arrays
data = np.ascontiguousarray( data[:,np.newaxis, np.newaxis, :] )
labels = np.ascontiguousarray( labels )

#Load data in the input
solver.net.set_input_arrays(data, labels)

#Solve
solver.solve()
