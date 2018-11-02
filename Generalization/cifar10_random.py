import tensorflow as tf
import numpy as np
from keras.datasets import cifar10

image_size = 32
crop_size = 2
image_channel = 3
learning_rate = 0.1
batch_size = 2000
epochs = 10
class_num = 10

def crop_image(images, c):
    images = images[:, c:image_size-c, c:image_size-c]
    return images

def scale_pixel_value(images):
    return images/255.

def onehot(label):
    onehot_label = []
    for l in label:
        lt = [0]*class_num
        lt[l[0]-1] = 1
        onehot_label.append(lt)
    return np.array(onehot_label)

def partially_corrupted_label(label, probability):
    pcl = []
    for l in label:
        if np.random.random_sample() < probability:
            pcl.append([np.random.randint(class_num)])
        else:
            pcl.append([l[0]])
    return pcl

def random_label(label):
    return np.reshape(np.random.randint(class_num, size=len(label)), [-1, 1])

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = scale_pixel_value(np.reshape(crop_image(x_train, crop_size), 
                                       [-1, (image_size - crop_size**2)**2 * image_channel]))
num_train = x_train.shape[0]
image_size = image_size - crop_size**2
print (x_train.shape, y_train.shape)

n_input_size = image_size**2 * image_channel
n_hidden_1 = 16
n_hidden_2 = 16
n_hidden_3 = 16
n_output_size = class_num
sigma = 0.01

weights = {
    'W1': tf.Variable(tf.random_normal([n_input_size, n_hidden_1], stddev=sigma)),
    'W2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=sigma)),
    'W3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=sigma)),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_output_size], stddev=sigma))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_output_size]))

}

def multilayer_perceptron_1layer(x, weights, biases):
    layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['W1']), biases['b1']))
    output_layer = tf.nn.relu(tf.add(tf.matmul(layer1, weights['out']), biases['out']))
    return output_layer
    
def multilayer_perceptron_3layer(x, weights, biases):
    layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['W1']), biases['b1']))
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['W2']), biases['b2']))
    layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, weights['W3']), biases['b3']))
    output_layer = tf.nn.relu(tf.add(tf.matmul(layer3, weights['out']), biases['out']))
    return output_layer

def train(X_train, Y_train, num_layer):
    x = tf.placeholder(tf.float32, [None, n_input_size], name='input')
    y = tf.placeholder(tf.float32, [None, n_output_size], name='output')
    pred = multilayer_perceptron_1layer(x, weights, biases)
    if num_layer == 1:
        pred = multilayer_perceptron_1layer(x, weights, biases)
    elif num_layer == 3:
        pred = multilayer_perceptron_3layer(x, weights, biases)
    #loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    #optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    Y_train = onehot(Y_train)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        myIdx =  np.random.permutation(num_train)
        for epoch in range(epochs):
            num_batch = num_train / batch_size
            for i in range(int(num_batch)):
                x_batch = X_train[myIdx[i*batch_size:(i+1)*batch_size],:]
                y_batch = Y_train[myIdx[i*batch_size:(i+1)*batch_size],:]
                sess.run(optimizer, feed_dict={x: X_train, y: Y_train})
            loss_temp = sess.run(loss, feed_dict={x: X_train, y: Y_train}) 
            accuracy_temp = accuracy.eval({x: X_train, y: Y_train})
            print ("(epoch {})".format(epoch+1) )
            print ("[Loss / Tranining Accuracy] {:05.4f} / {:05.4f}".format(loss_temp, accuracy_temp))
            print (" ")
# train(x_train, y_train, 1)
# train(x_train,partially_corrupted_label(y_train,0.3),1)
train(x_train, random_label(y_train),1)