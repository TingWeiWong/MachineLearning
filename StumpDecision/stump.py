import numpy as np
import random
import sys

# Global Parameters
data_set_size = 2
tau = 0
experiment_epoch = 100

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

def find_Ein(data_set_size, tau):
	"""
	This function finds the Error for Ein
	 Need to regenerate data for X_list every loop
	- Input:
	- Output:
		* Ein: Ein for that specific experiment
		* theta: s*sign(x-theta), the value for theta
		* sign_value: s = +1 or -1
	"""
	x_Ein = numpy.random.uniform(-1.0,1.0,data_set_size)
	x_Ein = numpy.sort(x_Ein) # O(nlogn)
	threshold_list = [-1]
	y_list = label_with_sign(x_Ein,data_set_size,tau)
	y_list = flipping(y_list,data_set_size,tau)

	# Append threshold list
	for i in range(data_set_size-1):
		mid_section = (x[i]+x[i+1])/2
		threshold_list.append(mid_section)

	# Local Parameters
	final_Ein = float('inf')

	sign_value = 1
	threshold = 0.0

	# Loop over all possible intervals
	for i in range(data_set_size):
		positive_Ein = 0.0 # Ein case for setting s = 1
		negative_Ein = 0.0 # Ein case for setting s = -1
		for j in range(data_set_size):
			data_correspondence = (x[j] - threshold_list[i]) * y[j]
			if data_correspondence <= 0:
				positive_Ein += 1
			else:
				negative_Ein += 1

		# Normalize Ein 
		positive_Ein = positive_Ein / data_set_size
		negative_Ein = negative_Ein / data_set_size

		# Return s = 1 or s = -1
		if positive_Ein <= negative_Ein:
			if positive_Ein < final_Ein:
				threshold = threshold_list[i]
				final_Ein = positive_Ein
				sign_value = 1

		else:
			if negative_Ein < final_Ein:
				threshold = threshold_list[i]
				final_Ein = negative_Ein
				sign_value = -1

	# Return values
	return final_Ein, threshold, sign_value




# x = [1,2,3,4,5,6,7]
# for i in range(len(x)):
# 	x[i] = -x[i]

# print ("New x = ",x)
