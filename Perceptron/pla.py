from random import seed
from random import randint
import numpy as np
import os

# curDir = os.getcwd()
raw = [line for line in open("hw1_train.dat").readlines()]
x = []; y = []
for d in raw:
	x_n = d.split("\t")
	if x_n[-1][-1]=="\n": x_n[-1] = x_n[-1][:-1]
	x_n = [1.0] + [float(s) for s in x_n]
	x.append(x_n[:-1])
	y.append(x_n[-1])
x = np.array(x)
y = np.array(y)


def sign(in_int):
	if in_int>0: return 1
	else: return -1

def PLA(w, x, y, stop):
	t = 0; boo = 0
	while boo < stop:
		n = randint(0, N-1)
		boo += 1
		if y[n]*np.dot(w, x[n])<=0:
			w = w + y[n]*x[n]
			t += 1
			boo = 0
	return w, t


iter = 1000
N = x.shape[0]
w = np.zeros(x.shape[1])
wPLA = np.zeros(x.shape[1])

# for 16.
upNums = []
# for 17.
w0s = []
# for 18.
upNums_18 = []
x_18 = np.array([[10, *xn[1:]] for xn in x])
# for 19.
upNums_19 = []
x_19 = np.array([[0, *xn[1:]] for xn in x])
# for 20.
upNums_20 = []
x_20 = np.array([[0, *xn[1:]] for xn in x])/4

for i in range(iter):
	seed(i+1)
	wPLA, upNum = PLA(w, x, y, 5*N)
	upNums.append(upNum)
	w0s.append(wPLA[0])
	
	wPLA_18, upNum_18 = PLA(w, x_18, y, 5*N)
	upNums_18.append(upNum_18)
	wPLA_19, upNum_19 = PLA(w, x_19, y, 5*N)
	upNums_19.append(upNum_19)
	wPLA_20, upNum_20 = PLA(w, x_20, y, 5*N)
	upNums_20.append(upNum_20)

upNumMed = np.median(upNums)
w0Med = np.median(w0s)
upNumMed_18 = np.median(upNums_18)
upNumMed_19 = np.median(upNums_19)
upNumMed_20 = np.median(upNums_20)

print("16. Median number of updates: {}".format(upNumMed))
print("17. Median of w_0: {}".format(w0Med))
print("18. Median number of updates: {}".format(upNumMed_18))
print("19. Median number of updates: {}".format(upNumMed_19))
print("20. Median number of updates: {}".format(upNumMed_20))
