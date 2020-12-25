from svmutil import *
y_train, x_train = svm_read_problem('../data/satimage.scale')
y_test, x_test = svm_read_problem('../data/satimage.scale.t')


C_list = [0.01, 0.1, 1, 10, 100]

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

for C_value in C_list:
	print ("C_value = ",C_value)
	param = ("-s 0 -t 2 -g 10 -c {}".format(C_value))
	model = svm_train(y_train,x_train,param)
	p_labels, p_acc, p_vals = svm_predict(y_test, x_test, model)
	print ("p_acc = ",p_acc)

