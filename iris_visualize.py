import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
k = 3
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
def PCA(data, k=3):
	# preprocess the data
	X = torch.from_numpy(data)
	X_mean = torch.mean(X,0)
	X = X - X_mean.expand_as(X)

	# svd
	U,S,V = torch.svd(torch.t(X))
	return torch.mm(X,U[:,:k])
X_PCA = PCA(X)
plt.figure()

for i, target_name in enumerate(iris.target_names):
	plt.scatter(X_PCA[y == i, 0], X_PCA[y == i, 1], label=target_name)

plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()