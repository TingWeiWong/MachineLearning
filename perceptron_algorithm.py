from random import seed
from random import randint
import numpy as np


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
		x_part = new_split_data[:-1]
		y_part = new_split_data[-1]
		# print ("x, y = ",x_part, y_part)
		x_list.append(x_part)
		y_list.append(y_part)

	x_list, y_list = np.array(x_list), np.array(y_list)

	return x_list, y_list

def perceptron_main(x, y, weight, max_count):
	"""
	This function executes the main PLA algorithm
	"""
	time_step, counter = 0 

	# Find random number between 0 and N-1
	while counter < max_count:
		random_number = randint(0,N-1)
		counter += 1
		# Find incorrect label
		value = y[n] * np.dot(weight,x[n])
		if value <= 0:
			weight += y[n] * x[n] # Vector addition
			time_step += 1
			counter = 0 

	return weight, time_step







content_list = read_file("hw1_train.dat")
x_list, y_list = split_data(content_list)
print ("x_list, y_list = ",len(x_list), len(y_list))
# print ("content_list = ",content_list)
# for index in content_list:
# 	print ("len of single line = ",len(index))