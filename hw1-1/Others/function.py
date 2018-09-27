import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

n = 20
x = np.random.uniform(-15, 15, size = n)
y = x**2 + 2*np.random.randn(n, )
X = np.reshape(x ,[n, 1]) 
y = np.reshape(y ,[n ,])


clf = MLPRegressor(alpha=0.001, hidden_layer_sizes = (10,), max_iter = 50000, 
                 activation = 'relu', verbose = 'True', learning_rate = 'adaptive')
a = clf.fit(X, y)

x_ = np.linspace(-20,20, 10)

pred_x = np.reshape(x_, [10, 1])
pred_y = clf.predict(pred_x)
fig = plt.figure() 
plt.plot(x_, x_**2, color = 'r')
plt.plot(pred_x, pred_y, '-')
plt.show()