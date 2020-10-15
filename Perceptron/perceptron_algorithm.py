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
		# x_part = new_split_data[:-1]
		new_split_data = [1.0] + [float(element) for element in new_split_data]
		# print ("x_part = ",x_part)
		# y_part = new_split_data[-1]
		# print ("x, y = ",x_part, y_part)
		x_list.append(new_split_data[:-1])
		y_list.append(new_split_data[-1])

	x_list, y_list = np.array(x_list), np.array(y_list)

	return x_list, y_list

def perceptron_main(x, y, weight, max_count):
	"""
	This function executes the main PLA algorithm
	"""
	time_step, counter = 0, 0

	# Find random number between 0 and N-1
	while counter < max_count:
		random_number = randint(0,N-1)
		counter += 1
		# Find incorrect label
		value = y[random_number] * np.dot(weight,x[random_number])
		if value <= 0:
			weight = weight + y[random_number] * x[random_number] # Vector addition
			time_step += 1
			counter = 0 

	return weight, time_step

# GLOBAL PARAMS
iteration = 1000
content_list = read_file("hw1_train.dat")
x_list, y_list = split_data(content_list)

N = x_list.shape[0]
max_count = 5*N
init_weight = np.zeros(x_list.shape[1])
returned_weight = np.zeros(x_list.shape[1])
print ("x_list = ",x_list)

# Different question setup
time_step_16_list = []
weight_17_list = []
time_step_18_list = []
x_list_18 = np.array([[10, *xn[1:]] for xn in x_list])
time_step_19_list = []
x_list_19 = np.array([[0, *xn[1:]] for xn in x_list])
time_step_20_list = []
x_list_20 = np.array([[0, *xn[1:]] for xn in x_list]) / 4


# Main Loop 
for i in range(iteration):
	seed(i+1)
	returned_weight, time_step_16 = perceptron_main(x_list,y_list,init_weight,max_count)
	time_step_16_list.append(time_step_16)
	weight_17_list.append(returned_weight[0])
	
	weight_18, time_step_18 = perceptron_main(x_list_18,y_list,init_weight,max_count)
	time_step_18_list.append(time_step_18)
	weight_19, time_step_19 = perceptron_main(x_list_19,y_list,init_weight,max_count)
	time_step_19_list.append(time_step_19)
	weight_20, time_step_20 = perceptron_main(x_list_20,y_list,init_weight,max_count)
	time_step_20_list.append(time_step_20)

Med16 = np.median(time_step_16_list)
MedWeight = np.median(weight_17_list)
Med18 = np.median(time_step_18_list)
Med19 = np.median(time_step_19_list)
Med20 = np.median(time_step_20_list)

print("16. : {}".format(Med16))
print("17. : {}".format(MedWeight))
print("18. : {}".format(Med18))
print("19. : {}".format(Med19))
print("20. : {}".format(Med20))

# print ("content_list = ",content_list)
# for index in content_list:
# 	print ("len of single line = ",len(index))