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

	weight_X_dot = np.dot(w,x)

	if mode == "zero":
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









if __name__ == "__main__":
	content_list = read_file("hw3_train.dat")
	x_list, y_list = split_data(content_list)
	# print ("x_list = ",x_list)
	# print ("y_list = ",y_list)



