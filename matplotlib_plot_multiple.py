import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import math

t = np.arange(0.,5.,0.2)
with open("Testing.png") as graph:
	for i in range(3):
		plt.plot(t,t+i,'r')
		plt.savefig()
