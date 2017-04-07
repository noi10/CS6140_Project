import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import json
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
    trainLabels = np.transpose(np.array(json.load(infile)).astype(np.float32))
infile.close()

with open('./txtdata/yt8m_100/svm_validate_labels.txt','r') as infile:
    validateLabels = np.transpose(np.array(json.load(infile)).astype(np.float32))
infile.close()

with open('./txtdata/yt8m_100/test_labels.txt','r') as infile:
    testLabels = np.array(json.load(infile)).astype(np.float32)
infile.close()


# Create graph
sess = tf.Session()

x_vals = trainFeatures
y_vals = trainLabels

batch_size = 2048
num_features = 1152
num_labels = 10
regulation_rate = 5e3

x_data = tf.placeholder(shape=[None, 1152], dtype=tf.float32)
y_target = tf.placeholder(shape=[10, None], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, 1152], dtype=tf.float32)

b = tf.Variable(tf.random_normal(shape=[num_labels, batch_size]))
zeros = tf.zeros(shape=[num_labels, batch_size])

#gamma = tf.constant(-100.0)
#dist = tf.reduce_sum(tf.square(x_data), 1)
#dist = tf.reshape(dist, [-1,1])
#sq_dists = tf.multiply(tf.constant(2, dtype=tf.float32), tf.matmul(x_data, tf.transpose(x_data)))
#my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))
my_kernel = tf.matmul(x_data, tf.transpose(x_data))
def reshape_matmul(mat):
    v1 = tf.expand_dims(mat, 1)
    v2 = tf.reshape(v1, [10, batch_size, 1])
    return(tf.matmul(v2, v1))

first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = reshape_matmul(y_target)

second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)),[1,2])
#second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))
#loss += regulation_rate * tf.nn.l2_loss(b)



pred_kernel = tf.matmul(x_data, tf.transpose(prediction_grid))
#rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1),[-1,1])
#rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1),[-1,1])
#pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(tf.constant(2, dtype=tf.float32), tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
#pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

prediction_output = tf.matmul(tf.multiply(y_target,b), pred_kernel)
prediction = tf.arg_max(prediction_output, 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target,0)), tf.float32))


#base_learning_rate = 0.05
#learning_rate_decay_examples = 2000
#learning_rate_decay = 0.2
#global_step = tf.Variable(0, trainable=False)
#learning_rate = tf.train.exponential_decay(
#  base_learning_rate,
#  global_step,
#  learning_rate_decay_examples,
#  learning_rate_decay,
#  staircase=True)
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)



my_opt = tf.train.AdamOptimizer(0.01)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
batch_accuracy = []
for i in range(2500):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = y_vals[:,rand_index]
    #rand_x = x_vals[:batch_size]
    #rand_y = y_vals[:,:batch_size]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    
    acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid:rand_x})
    batch_accuracy.append(acc_temp)
    
    if (i+1)%50==0:
        print('Step #' + str(i+1))
        print('Loss = ' + str(temp_loss))
        print('Mini Batch accuracy = ' + str(acc_temp))
        acc_va = sess.run(tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(validateLabels,0)), tf.float32)) ,feed_dict={x_data: rand_x, y_target: rand_y,prediction_grid: validateFeatures})
        print('Validation accuracy = ' + str(acc_va))

acc_test = sess.run(tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(testLabels,1)), tf.float32)) ,feed_dict={x_data: rand_x, y_target: rand_y,prediction_grid: testFeatures})
print('Test accuracy = ' + str(acc_test))      

