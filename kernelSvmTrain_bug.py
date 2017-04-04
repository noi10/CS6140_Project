# Multi-class (Nonlinear) SVM Example
#----------------------------------
#
# This function wll illustrate how to
# implement the gaussian kernel with
# multiple classes on the iris dataset.
#
# Gaussian Kernel:
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)
#
# X : (Sepal Length, Petal Width)
# Y: (I. setosa, I. virginica, I. versicolor) (3 classes)
#
# Basic idea: introduce an extra dimension to do
# one vs all classification.
#
# The prediction of a point will be the category with
# the largest margin or distance to boundary.

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
from tensorflow.python.framework import ops
ops.reset_default_graph()


with open('./txtdata/train_features.txt','r') as infile:
    trainFeatures = np.array(json.load(infile)).astype(np.float32)
infile.close()

with open('./txtdata/validate_features.txt','r') as infile:
    validateFeatures = np.array(json.load(infile)).astype(np.float32)
infile.close()

with open('./txtdata/test_features.txt','r') as infile:
    testFeatures = np.array(json.load(infile)).astype(np.float32)
infile.close()

with open('./txtdata/svm_train_labels.txt','r') as infile:
    trainLabels = np.array(json.load(infile)).astype(np.float32)
infile.close()

with open('./txtdata/svm_validate_labels.txt','r') as infile:
    validateLabels = np.array(json.load(infile)).astype(np.float32)
infile.close()

with open('./txtdata/test_labels.txt','r') as infile:
    testLabels = np.array(json.load(infile)).astype(np.float32)
infile.close()
    
# Create graph
sess = tf.Session()


x_vals = trainFeatures
y_vals = np.transpose(trainLabels)

# Declare batch size
batch_size = 256
num_features = 1152
num_labels = 10

# Initialize placeholders
x_data = tf.placeholder(shape=[None, 1152], dtype=tf.float32)
y_target = tf.placeholder(shape=[10, None], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, 1152], dtype=tf.float32)

# Create variables for svm
#b = tf.Variable(tf.random_normal(shape=[10, batch_size]))
b = tf.Variable(tf.random_normal(shape=[10, 1]))
#b = tf.Variable(tf.zeros(shape=[10, 1]))

# Gaussian (RBF) kernel
gamma = tf.constant(-50.0)
dist = tf.reduce_sum(tf.square(x_data), 1)
dist = tf.reshape(dist, [-1,1])
sq_dists = tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))
my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))
#my_kernel = tf.matmul(x_data, tf.transpose(x_data))
# Declare function to do reshape/batch multiplication

def reshape_matmul(mat):
    v1 = tf.expand_dims(mat, 1)
    v2 = tf.reshape(v1, [10, batch_size, 1])
    return(tf.matmul(v2, v1))

# Compute SVM Model
#bb = tf.reshape(b, [-1, 1])
first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = reshape_matmul(y_target)

second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)),[1,2])
loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

#pred_kernel = tf.matmul(x_data, tf.transpose(x_data))

def model(x_data, prediction_grid, y_target):
    # Gaussian (RBF) prediction kernel
    rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1),[-1,1])
    #return rA
    rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1),[-1,1])
    #return rB
    pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
    #return pred_sq_dist
    pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))
    prediction_output = tf.matmul(tf.multiply(y_target, b), pred_kernel)
    #return prediction_output
    prediction = tf.arg_max(prediction_output-tf.expand_dims(tf.reduce_mean(prediction_output,1), 1), 0)
    return prediction

def accuracy(prediction, labels):
    acc = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(labels, 0)), tf.float32))
    return 100*tf.to_float(acc, name='ToFloat')


global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1

learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 800, 0.7, staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
#optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

tf_train_dataset = tf.constant(trainFeatures)
tf_valid_dataset = tf.constant(validateFeatures)
validateLabels = np.transpose(validateLabels)
tf_valid_labels = tf.constant(validateLabels)
tf_test_dataset = tf.constant(testFeatures)
testLabels = np.transpose(testLabels)
tf_test_labels = tf.constant(testLabels)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

train_prediction = model(x_data, prediction_grid, y_target)
valid_prediction = model(tf_train_dataset, tf_valid_dataset, tf_valid_labels)
test_prediction = model(tf_train_dataset, tf_test_dataset, tf_test_labels)

# Training loop
for step in range(1):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = y_vals[:,rand_index]
    #print(rand_y)

    feed_dict = {x_data: rand_x, prediction_grid: rand_x, y_target: rand_y}
    _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict = feed_dict)
    #sess.run(valid_prediction)
    if (step%10==0):
        print("Minibatch loss at step %d: %f" % (step,l))
        acc = sess.run( accuracy(predictions, rand_y) )
        print("Minibatch accuracy: %.1f%%" % acc)
        acc = sess.run( accuracy(valid_prediction, validateLabels) )
        print("Validation accuracy: %.1f%%" % acc)
acc = sess.run( accuracy(test_prediction, testLabels) )
print("Test accuracy: %.1f%%" % acc)
