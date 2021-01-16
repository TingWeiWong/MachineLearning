from trees import *
import random 

forest_number = 100
replacement = 500

forest = [0] * forest_number

training_data = load_data("data/hw6_train.dat")
testing_data = load_data("data/hw6_test.dat")

contacted = [False] * 1000

for i in range(forest_number):
	x_random = []
	for j in range(replacement):
		randnum = random.randint(0,999)
		contacted[randnum] = True
		x_random.append(training_data[randnum])
	forest[i] = Construction(x_random)

oob_data_list = []
for i in range(1000):
	if contacted[i] == False:
		oob_data_list.append(training_data[i])

if oob_data_list == []:
	mode = 0
else:
	mode = 1

print ("oob_data_list = ",oob_data_list)


y_list = []
for i in range(len(oob_data_list)):
	temp = 0
	for j in range(forest_number):
		temp += get_label(Sort_data(oob_data_list[i], forest[j]))

	if (temp >= 0):
		y_list.append(1.0)
	else:
		y_list.append(-1.0)

error = 0
if mode == 1:
	for j in range(len(oob_data_list)):
		if (oob_data_list[j][-1] != y_list[j]):
			error += 1
else:
	print ("Empty")


error = error / (1000)
print ("error = ",error)




