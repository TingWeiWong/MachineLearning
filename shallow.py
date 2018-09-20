import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.optim as optim

# Function that we are trying to approximate
def sinc(x):
	return np.sinc(x)
def sinc_derivative(x):
	return (np.cos(np.pi*x)-sinc(x))/x
	
print (sinc_derivative(3))

