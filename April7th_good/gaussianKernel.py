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


with open('./txtdata/yt8m_100/train_features.txt','r') as infile:
    trainFeatures = np.array(json.load(infile)).astype(np.float32)
infile.close()

with open('./txtdata/yt8m_100/validate_features.txt','r') as infile:
    validateFeatures = np.array(json.load(infile)).astype(np.float32)
infile.close()

with open('./txtdata/yt8m_100/test_features.txt','r') as infile:
    testFeatures = np.array(json.load(infile)).astype(np.float32)
infile.close()

with open('./txtdata/yt8m_100/svm_train_labels.txt','r') as infile:
    trainLabels = np.array(json.load(infile)).astype(np.float32)
infile.close()

with open('./txtdata/yt8m_100/validate_labels.txt','r') as infile:
    validateLabels = np.transpose(np.array(json.load(infile)).astype(np.float32))
infile.close()

with open('./txtdata/yt8m_100/test_labels.txt','r') as infile:
    testLabels = np.transpose(np.array(json.load(infile)).astype(np.float32))
infile.close()
    
# Create graph
sess = tf.Session()


x_vals = trainFeatures
y_vals = np.transpose(trainLabels)

# Declare batch size
batch_size = 2048

# Initialize placeholders
x_data = tf.placeholder(shape=[None, 1152], dtype=tf.float32)
y_target = tf.placeholder(shape=[10, None], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, 1152], dtype=tf.float32)

# Create variables for svm
b = tf.Variable(tf.random_normal(shape=[10, batch_size]))
#b = tf.Variable(tf.random_normal(shape=[10]))

# Gaussian (RBF) kernel
gamma = tf.constant(-0.05)
dist = tf.reduce_sum(tf.square(x_data), 1)
dist = tf.reshape(dist, [-1,1])
sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))
#my_kernel = tf.matmul(x_data, tf.transpose(x_data))
# Declare function to do reshape/batch multiplication
def reshape_matmul(mat):
    v1 = tf.expand_dims(mat, 1)
    v2 = tf.reshape(v1, [10, batch_size, 1])
    return(tf.matmul(v2, v1))

# Compute SVM Model
first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = reshape_matmul(y_target)

second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)),[1,2])
loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

# Gaussian (RBF) prediction kernel
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1),[-1,1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1),[-1,1])
pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))
#pred_kernel = tf.matmul(x_data, tf.transpose(x_data))

prediction_output = tf.matmul(tf.multiply(y_target,b), pred_kernel)
prediction = tf.arg_max(prediction_output, 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target,0)), tf.float32))

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.01

learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 300, 0.5, staircase=True)

#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
for i in range(2500):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = y_vals[:,rand_index]
    #print(rand_y)
    sess.run(optimizer, feed_dict={x_data: rand_x, y_target: rand_y})
    
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    
    acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,
                                             y_target: rand_y,
                                             prediction_grid:rand_x})
    #train_acc = sess.run(accuracy, feed_dict = {x_data: x_vals,
    #                                            y_target: y_vals,
    #                                            prediction_grid: x_vals})
    if (i+1)%100==0:
        print('Step #' + str(i+1))
        print('Loss = ' + str(temp_loss))
        print('batch_accuracy = ' + str(acc_temp))
    #    print('train_accuracy = ' + str(train_acc))
        valid_prediction = sess.run(prediction, feed_dict={x_data: rand_x,
                                             y_target: rand_y,
                                             prediction_grid:validateFeatures})
        valid_acc = sess.run(tf.reduce_mean(tf.cast(tf.equal(valid_prediction, tf.argmax(validateLabels,0)), tf.float32)))
        print('validate_accuracy = ' + str(valid_acc))
test_prediction = sess.run(prediction, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid:testFeatures})
test_acc = sess.run(tf.reduce_mean(tf.cast(tf.equal(test_prediction, tf.argmax(testLabels, 0)), tf.float32)))
print('test_accuray =' + str(test_acc))        
