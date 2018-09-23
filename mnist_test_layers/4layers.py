import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.preprocessing import *
from sklearn.cross_validation import *
from sklearn.metrics import *
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
	n_hidden = 4	
	# Build a simple neural network.
	model = Sequential()
	model.add(Dense(input_dim = x_train.shape[1], output_dim = n_hidden))
	model.add(Activation('relu'))
	n=550
	model.add(Dense(n))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('relu'))
	model.add(Dense(output_dim = 10))
	model.add(Activation('softmax'))
	sgd = SGD(lr=0.2, decay=1e-7, momentum=0.1, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
	model.summary()
	model.fit(x_train, y_train, epochs = 10, batch_size = 10, verbose = 1, validation_split = 0.05)
	score = model.evaluate(x_train,y_train)
	print('\nTrain Acc for:',(n_hidden),"layers", score[1])
	score = model.evaluate(x_test,y_test)
	print('\nTest Acc:', score[1])