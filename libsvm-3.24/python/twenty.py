from svmutil import *
import random

y_train, x_train = svm_read_problem('../data/satimage.scale')
number_of_data = len(y_train)
print ("data length = ",number_of_data)
gamma_count = {0.1:0, 1:0, 10:0, 100:0, 1000:0}
gamma_list = [0.1, 1, 10, 100, 1000]

# 6 versus not 6
for i in range(len(y_train)):
	if y_train[i] == 6:
		y_train[i] = 1
	else:
		y_train[i] = -1

loop_experiment = 1000

for i in range(loop_experiment):
	# Sample 200 for Eval
	data = list(range(number_of_data))
	# print ("data = ",data)
	random.shuffle(data)
	# print ("total_data = ",data)
	eval_data_list, train_data_list = data[:200], data[200:]
	# print ("eval_data_list = ",eval_data_list)
	new_x_train, new_x_eval = [], []
	new_y_train, new_y_eval = [], []
	for j in range(number_of_data):
		if j in eval_data_list:
			new_x_eval.append(x_train[j])
			new_y_eval.append(y_train[j])
		else:
			new_x_train.append(x_train[j])
			new_y_train.append(y_train[j])

	max_acc_gamma, max_acc_value = "null", 0
	for gamma_value in gamma_list:
		param = ("-s 0 -t 2 -g {} -c 0.1".format(gamma_value))
		model = svm_train(new_y_train,new_x_train,param)
		p_labels, p_acc, p_vals = svm_predict(new_y_eval, new_x_eval, model)
		# print ("p_acc = ",p_acc[0])		
		if p_acc[0] > max_acc_value:
			max_acc_gamma = gamma_value
			max_acc_value = p_acc[0]

	gamma_count[max_acc_gamma] += 1
print ("gamma_count = ",gamma_count)


