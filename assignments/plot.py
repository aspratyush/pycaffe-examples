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
