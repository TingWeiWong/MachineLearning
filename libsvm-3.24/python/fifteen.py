from svmutil import *
y, x = svm_read_problem('../data/satimage.scale')

# 3 versus not 3
for i in range(len(y)):
	if y[i] == 3:
		y[i] = 1
	else:
		y[i] = -1

model = svm_train(y,x,'-s 0 -t 0 -d 1 -c 10')
support_vectors = model.get_SV()
support_vector_coefficients = model.get_sv_coef()
weight = {}

for i in range(len(support_vectors)):
	sup_dic = support_vectors[i]
	coef = support_vector_coefficients[i][0]
	for index in sup_dic:
		print ("sup_dic[index] = ",sup_dic[index])
		print ("coef = ",coef)
		if index not in weight:
			weight[index] = float(sup_dic[index]) * coef
		else:
			weight[index] += float(sup_dic[index]) * coef

ans = 0

print ("weight = ",weight)

for index in weight:
	ans += weight[index] ** 2

print ("ans = ",ans)

# for i in range()