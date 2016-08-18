import matplotlib.pyplot as plt
import numpy as np
import subprocess

#run script to analyze loss
print "start plotting..."
subprocess.call("./plotting.sh", shell=True)
#plot
train_loss = np.genfromtxt('./solve1.log', delimiter=',', names=['x', 'y'])
plt.plot(train_loss['x'], train_loss['y'],color='r'), plt.show()
