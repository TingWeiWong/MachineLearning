from random import randint
import numpy as np
import os

def read_file(file_name):
	"""
	This function opens the data file and loads all (x,y) into list
	"""
	with open(file_name) as read_file:
		content_list =[line.rstrip('\n') for line in read_file]
	return content_list

def split_data(content_list):
	"""
	This function splits content list into x,y parts 
	"""
	x_list, y_list = [], []
	for single_point in content_list:
		new_split_data = single_point.split("\t")
		# print ("new_split_data = ",new_split_data)
		# x_part = new_split_data[:-1]
		new_split_data = [1.0] + [float(element) for element in new_split_data]
		# print ("x_part = ",x_part)
		# y_part = new_split_data[-1]
		# print ("x, y = ",x_part, y_part)
		x_list.append(new_split_data[:-1])
		y_list.append(new_split_data[-1])

	x_list, y_list = np.array(x_list), np.array(y_list)

	return x_list, y_list


def error_function(x, y, weight, mode):
	"""
	This function returns Average error measure for different modes
		- mode:
			* zero: for zero error
			* squared: for squared error
			* cross: cross-entropy error
	"""
	# All requires W.transpose() X

	weight_X_dot = np.dot(x,weight)

	if mode == "zero":
		# Numpy multiply is an element-wise operation!!!
		result = np.multiply(y,weight_X_dot) <= 0

	elif mode == "squared":
		result = np.square(y - weight_X_dot)

	elif mode == "cross":
		exponent = -np.multiply(y,weight_X_dot)
		result = np.log(1 + np.exp(exponent))

	else:
		print ("Mode not supported!")
		return False

	return result.mean()


def linear_regression(x, y):
	"""
	This function computes the linear regression result 
	using the pseudo inverse formula.
	- Input:
		* x, y
	- Returns:
		* W = pseudo inverse of X ...
	"""
	x_transpose = x.transpose()
	symmetric_matrix = np.dot(x_transpose,x)

	# Matrix may be invertible if Positive definite
	try:
		inverse_symmetric = np.linalg.inv(symmetric_matrix)
	except:
		print ("Matrix not invertible, using pseudo inverse")
		inverse_symmetric = np.linalg.pinv(symmetric_matrix)

	result_x = np.dot(inverse_symmetric,x_transpose)

	optimal_weight = np.dot(result_x,y)

	return optimal_weight


def SGD_algorithm(x, y, mode, initial_weight = "zero", learning_rate = 1E-3, experiment_loop = 1000, threshold_value = 1.01):
	"""
	This function implements the SGD update for linear regression
	and logistic regression specified by mode
	"""

	# Specify initial weight Xw -> shape of w should be x.shape[1] x 1
	data_num, dimension = x.shape[0], x.shape[1]
	if initial_weight == "zero":
		initial_weight = np.zeros(dimension)

	if mode == "linear":
		w_lin = linear_regression(x, y)
		Error_upper_bound = threshold_value * error_function(x,y,w_lin,mode="squared")
		Ein_mean = float("inf")
		iteration = 0
		while  Ein_mean > Error_upper_bound:
			# Generate seed
			stochastic_value = randint(0,data_num-1)
			error_vector = y - np.dot(x, initial_weight)
			initial_weight += 2 * learning_rate * error_vector[stochastic_value] * x[stochastic_value]
			Ein_mean = np.square(error_vector).mean()
			iteration += 1
		return iteration


	elif mode == "logistic":
		pass
	else:
		print ("Mode not supported!")
		return False




if __name__ == "__main__":
	train_content_list = read_file("hw3_train.dat")
	train_x_list, train_y_list = split_data(train_content_list)
	# optimal_weight = linear_regression(train_x_list,train_y_list)
	# squared_error = error_function(train_x_list, train_y_list, optimal_weight, mode = "squared")
	# print ("squared_error for 14 = ",squared_error)
	iteration_number = 1000
	count = 0 
	for i in range(iteration_number):
		count += SGD_algorithm(train_x_list,train_y_list,mode="linear")
	result = count / iteration_number
	print ("result = ",result)


	# print ("train_x_list = ",train_x_list)
	# print ("train_y_list = ",train_y_list)



