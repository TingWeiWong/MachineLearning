from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

num_classes = 10
img_size = 32 # cifar10 size = 32x32x3


def load_data():
	# load cifar10 data
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	# preprocess data, let pixel between 0~1
	x_train = x_train.reshape((x_train.shape[0], img_size * img_size*3))
	x_train = x_train.astype('float32') / 255.

	x_test = x_test.reshape((x_test.shape[0], img_size * img_size*3))
	x_test = x_test.astype('float32') / 255.

	y_train = np_utils.to_categorical(y_train, num_classes)
	y_test = np_utils.to_categorical(y_test, num_classes)

	return x_train, y_train, x_test, y_test


if __name__ == '__main__':
	x_train, y_train, x_test, y_test = load_data()
	print ((y_train)[0])
	print ((y_test)[0])