import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train),(x_test, y_test) = mnist.load_data()
img_size =28
x_train = x_train.reshape((x_train.shape[0], img_size * img_size))
x_train = x_train.astype('float32') / 255.

x_test = x_test.reshape((x_test.shape[0], img_size * img_size))
x_test = x_test.astype('float32') / 255.
y_train = np.random.permutation(y_train)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs = 10, validation_split = 0.2, shuffle = False)






# output score
score = model.evaluate(x_train,y_train)
print('\nTrain Acc:', score[1])
score = model.evaluate(x_test,y_test)
print('\nTest Acc:', score[1])



plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("MNIST_random_1000_epochs.png")
