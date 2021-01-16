from trees import *
import random 

forest_number = 100
replacement = 500

forest = [0] * forest_number

training_data = load_data("data/hw6_train.dat")
testing_data = load_data("data/hw6_test.dat")

contacted = []
for i in range(1000):
	contacted.append([False] * forest_number)

for i in range(forest_number):
	x_random = []
	for j in range(replacement):
		randnum = random.randint(0,999)
		contacted[randnum][i] = True
		x_random.append(training_data[randnum])
	forest[i] = Construction(x_random)

temp = 0
error = 0
for i in range(len(training_data)):
	for j in range(forest_number):
		if contacted[i][j] == False:
			temp += 1
			if (training_data[i][-1] != get_label(Sort_data(testing_data[i], forest[j]))):
				error += 1

print (float(error) / temp)






