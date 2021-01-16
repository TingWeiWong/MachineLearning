from trees import *
import random 

forest_number = 100
replacement = 500

forest = [0] * forest_number

training_data = load_data("data/hw6_train.dat")
testing_data = load_data("data/hw6_test.dat")

for i in range(forest_number):
	x_random = []
	for j in range(replacement):
		randnum = random.randint(0,999)
		x_random.append(training_data[randnum])
	forest[i] = Construction(x_random)


y_list = [0] * forest_number
for i in range(forest_number):
	y_list[i] = []

for j in range(forest_number):
	for i in range(len(testing_data)):
		predicted = get_label(Sort_data(testing_data[i], forest[j]))
		y_list[j].append(predicted)

error = 0
for j in range(forest_number):
	for i in range(len(testing_data)):
		if (y_list[j][i] != testing_data[i][-1]):
			error += 1

print ("error = ",error)




