import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Conv2D
from keras.optimizers import RMSprop
from keras import backend as K
import matplotlib.pyplot as plt
K.tensorflow_backend._get_available_gpus()
from keras.utils.np_utils import to_categorical
from keras import backend as k
import tensorflow
from matplotlib.pyplot import imshow
import numpy as np
import tensorflow as tf

#the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

batch_size = 128
num_classes = 10
epochs = 1


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(20, activation=LeakyReLU(), input_shape=(784,)))
model.add(Dense(20, activation=LeakyReLU()))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

#def get_gradient_norm_func(model):
weights = model.trainable_weights # weight tensors
weights = [weight for weight in weights ] # if model.get_layer(weight.name[:-2]).trainable filter down weights tensors to only ones which are trainable
gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors


outputTensor = model.output #Or model.layers[index].output
listOfVariableTensors = model.trainable_weights
gradients = k.gradients(outputTensor, listOfVariableTensors)
trainingExample = np.random.random((1,784))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
evaluated_gradients = sess.run(gradients,feed_dict={model.input:trainingExample})
print (evaluated_gradients)




# get_gradient = gradients
# plt.plot(get_gradient)
# plt.title('grad')
# plt.ylabel('grad')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
