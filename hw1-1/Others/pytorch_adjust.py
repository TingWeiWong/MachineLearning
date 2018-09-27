import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.optim as optim

# Single input output non-linear function to approximate
def function(x):
	return np.sinc(x)

def setupDataset():
	#Using Pandas columns 
	data = []
	
