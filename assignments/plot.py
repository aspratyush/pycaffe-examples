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
   
import matplotlib.pyplot as plt, mpld3
import numpy as np
import subprocess

#run script to analyze loss
print "start plotting..."
subprocess.call("./plotting.sh", shell=True)
#plot
train_loss = np.genfromtxt('./solve1.log', delimiter=',', names=['x', 'y'])
plt.plot(train_loss['x'], train_loss['y'], 'r--', label='training loss')
plt.xlabel('iterations'), plt.ylabel('loss')
plt.legend(shadow=True)
mpld3.show()
