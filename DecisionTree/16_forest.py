from trees import *
import random 

forest_number = 20
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

y_list = []
for i in range(len(training_data)):
	temp = 0
	for j in range(forest_number):
		temp += get_label(Sort_data(training_data[i], forest[j]))

	if (temp >= 0):
		y_list.append(1.0)
	else:
		y_list.append(-1.0)

error = 0
for j in range(len(training_data)):
	if (training_data[j][-1] != y_list[j]):
		error += 1

error = error / (1000)
print ("error = ",error)







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

error = error / (1000 * forest_number)
print ("error = ",error)




