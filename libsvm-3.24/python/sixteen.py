from svmutil import *

# k versus not k
for k in range(1,7):
	# 1 to 6
	y, x = svm_read_problem('../data/satimage.scale')
	print ("{} versus not {}".format(k,k))
	for i in range(len(y)):
		if y[i] == k:
			y[i] = 1
		else:
			y[i] = -1

	model = svm_train(y,x,'-s 0 -t 1 -d 2 -c 10 -r 1 -g 1')
	support_vectors = model.get_SV()
	support_vector_coefficients = model.get_sv_coef()
	p_labels, p_acc, p_vals = svm_predict(y, x, model)
	print ("p_acc = ",p_acc)
