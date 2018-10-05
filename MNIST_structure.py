from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

num_classes = 10
img_size = 28 # mnist size = 28*28


def load_data():
	# load mnist data
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# preprocess data, let pixel between 0~1
	x_train = x_train.reshape((x_train.shape[0], img_size * img_size))
	x_train = x_train.astype('float32') / 255.

	x_test = x_test.reshape((x_test.shape[0], img_size * img_size))
	x_test = x_test.astype('float32') / 255.

	y_train = np_utils.to_categorical(y_train, num_classes)
	y_test = np_utils.to_categorical(y_test, num_classes)

	return x_train, y_train, x_test, y_test


if __name__ == '__main__':
	x_train, y_train, x_test, y_test = load_data()
	print (len(y_test))
	print (len(y_train))
