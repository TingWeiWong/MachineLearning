from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as k
import numpy as np
import tensorflow as tf

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8,activation='relu'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
outputTensor = model.output #Or model.layers[index].output
listOfVariableTensors = model.trainable_weights
gradients = k.gradients(outputTensor, listOfVariableTensors)
trainingExample = np.random.random((1,8))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
evaluated_gradients = sess.run(gradients,feed_dict={model.input:trainingExample})
print (evaluated_gradients)