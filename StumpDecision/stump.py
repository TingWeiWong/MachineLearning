import numpy as np
import random
import sys

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
def flipping(y_list, data_set_size, tau):
	"""
	This function flips the y label by tau probability independently
	- Input:
		* y_list: sign(x) list
		* data_set_size: size of y_list
		* tau: percentage of flipping sign function
	"""
	if tau == 0.0 or tau == 0:
		return y_list
	elif tau == 0.1:
		for i in range(data_set_size):
			random_number = random.randint(0,9)
			if random_number == 1:
				y_list[i] = -y_list[i]
		return y_list

	else:
		sys.exit("tau not 0 nor 0.1 returning error")


# x = [1,2,3,4,5,6,7]
# for i in range(len(x)):
# 	x[i] = -x[i]

# print ("New x = ",x)
