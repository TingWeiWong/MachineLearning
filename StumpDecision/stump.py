import numpy as np
import random

# Global Parameters
data_set_size = 2
tau = 0

# Placeholders
Eout_minus_Ein = []


# Label y from x using sign function
def label_with_sign(x_list, data_set_size):
	"""
	This function applies the sign function on x_list,
	which is drawn from uniform distribution
	- Input:
		* x_list: list of uniform distribution [-1,+1]
		* data_size: size of x_list
	- Returns:
		* output_list: a same size list of x with y = sign(x)
	"""
	output_list = []

	for i in range(data_size):
		if x_list[i] > 0:
			output_list.append(1)
		else:
			output_list.append(-1)

	return output_list

# Flipping Function
def flipping(original_array, data_set_size, tau):
	"""
	This function flips the y label by tau probability independently
	- Input:
		* original_array:
		* 
	"""
	pass
