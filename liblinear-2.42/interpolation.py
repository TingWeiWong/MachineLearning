
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
		new_split_data = single_point.split(" ")
		# print ("new_split_data = ",new_split_data)
		# x_part = new_split_data[:-1]
		new_split_data = [float(element) for element in new_split_data]
		# print ("x_part = ",x_part)
		# y_part = new_split_data[-1]
		# print ("x, y = ",x_part, y_part)
		x_list.append(new_split_data[:-1])
		y_list.append(new_split_data[-1])

	x_list, y_list = np.array(x_list), np.array(y_list)

	# print ("x_list = ",x_list)
	return x_list, y_list


def loop_non_linear(train_x_list):
	"""
	This is the wrapper function for non linear phi interpolation
	"""
	non_linear_array = []
	data_len = len(train_x_list)

	for i in range(data_len):
		z_list = non_linear_transform(train_x_list[i])
		non_linear_array.append(z_list)

	return non_linear_array



def non_linear_transform(x_vector):
	"""
	This function expands the polynomials to max_power
	exp. (1,x1,x2,x1^2,x2^2,...,x1^Q,x2^Q)
	"""

	data_num = len(x_vector)
	# print ("data_num = ",data_num)

	z_list = [1.0]

	for index in x_vector:
		z_list.append(index)


	for i in range(data_num):
		for j in range(i,data_num):
			cross_term = x_vector[i] * x_vector[j]
			z_list.append(cross_term)

	# print ("z_list len = ",len(z_list))
	return z_list

def write_phi_transform(phi_matrix, file_path, train_y_list):
	"""
	This file writes back the non linear transform result
	"""
	N = len(train_y_list)
	print ("N = ",N)
	vector_len = len(phi_matrix[0])
	print ("phi_matrix_row = ",len(phi_matrix))
	print ("vector_len = ",vector_len)
	with open(file_path, 'w') as write_file:
		for i in range(N):
			if train_y_list[i] == 1:
				write_file.write("+1 ")
			else:
				write_file.write("-1 ")

			for j in range(vector_len):
				write_file.write("{}:{} ".format(j+1,phi_matrix[i][j]))

			write_file.write("\n")




if __name__ == "__main__":
	train_content_list = read_file("hw4_train.dat")
	train_x_list, train_y_list = split_data(train_content_list)
	# print ("train_x_list = ",train_x_list[0])
	# print ("train_y_list = ",train_y_list)
	phi_result = loop_non_linear(train_x_list)
	print ("phi_result = ",phi_result[0])
	write_phi_transform(phi_result,"phi_hw4_train.dat",train_y_list)





