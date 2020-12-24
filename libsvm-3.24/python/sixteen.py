from svmutil import *
y, x = svm_read_problem('../data/satimage.scale')

# k versus not k
for k in range(1,7):
	# 1 to 6
	print ("{} versus not {}".format(k,k))
	for i in range(len(y)):
		if y[i] == k:
			y[i] = 1
		else:
			y[i] = -1


model = svm_train(y,x,'-s 0 -t 0 -d 1 -c 10')
support_vectors = model.get_SV()
print ("support_vectors = ",support_vectors)
svm_type = model.get_svm_type()
print("svm_type = ",svm_type)
nr_class = model.get_nr_class()
print ("number of class = ",nr_class)
class_labels = model.get_labels()
print ("class_labels = ",class_labels)
sv_indices = model.get_sv_indices()
# print ("sv_indices = ",sv_indices)
nr_sv = model.get_nr_sv()
# print ("")
# support_vector_coefficients = model.get_sv_coef()
weight = {}

for dic in support_vectors:
	for index in dic:
		# print ("dic[index] = ",dic[index])
		if index not in weight:
			weight[index] = dic[index]
		else:
			weight[index] += dic[index]

ans = 0

print ("weight = ",weight)

for index in weight:
	ans += weight[index] ** 2

print ("ans = ",ans)

# for i in range()