# Copyright 2016 Pratyush Sahay
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#/bin/python

#Load a basic prototxt file and perform simple computations

import numpy as np
import caffe
import matplotlib.pyplot as plt

#Load the net
net = caffe.Net("./models/conv_prac2.prototxt", caffe.TRAIN)

#Load the image and typecast to np.array
inputData = np.array(
[[40,  6,  4],
[44, 10,  4],
[46, 12,  5],
[48, 14,  7],
[52, 16,  9],
[58, 18, 12],
[60, 22, 14],
[68, 24, 20],
[74, 26, 21],
[80, 32, 24]]
)

# Initialize memory data layer
data = np.ascontiguousarray(np.float32(inputData[:,np.newaxis, np.newaxis, 1:]))
predictions = np.ascontiguousarray(np.float32(inputData[:,0]))
#Initialize memorydata layer
net.set_input_arrays(data, predictions)

#net forward once
#net.forward()
