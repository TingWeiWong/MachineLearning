from svmutil import *
y_train, x_train = svm_read_problem('../data/satimage.scale')
y_test, x_test = svm_read_problem('../data/satimage.scale.t')


gamma_list = [0.1, 1, 10, 100, 1000]

# 6 versus not 6
for i in range(len(y_train)):
	if y_train[i] == 6:
		y_train[i] = 1
	else:
		y_train[i] = -1

for j in range(len(y_test)):
	if y_test[j] == 6:
		y_test[j] = 1
	else:
		y_test[j] = -1

for gamma_value in gamma_list:
	print ("gamma_value = ",gamma_value)
	param = ("-s 0 -t 2 -g {} -c 0.1".format(gamma_value))
	model = svm_train(y_train,x_train,param)
	p_labels, p_acc, p_vals = svm_predict(y_test, x_test, model)
	print ("p_acc = ",p_acc)

