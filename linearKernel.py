# Linear Kernel
# Linear SVCs are very powerful.  
# But sometimes the data are not very linear.  
# To this end, we can use the 'kernel trick' to map our data into 
# a higher dimensional space, where it may be linearly separable.  
# Doing this allows us to separate out non-linear classes.



import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import json
ops.reset_default_graph()

# data input
with open('./txtdata/yt8m_50_train_features.txt','r') as infile:
    trainFeatures = np.array(json.load(infile)).astype(np.float32)
infile.close()

with open('./txtdata/validate_features.txt','r') as infile:
    validateFeatures = np.array(json.load(infile)).astype(np.float32)
infile.close()

with open('./txtdata/test_features.txt','r') as infile:
    testFeatures = np.array(json.load(infile)).astype(np.float32)
infile.close()

with open('./txtdata/yt8m_50_svmtrain_labels.txt','r') as infile:
    trainLabels = np.transpose(np.array(json.load(infile)).astype(np.float32))
infile.close()

with open('./txtdata/svm_validate_labels.txt','r') as infile:
    validateLabels = np.transpose(np.array(json.load(infile)).astype(np.float32))
infile.close()

with open('./txtdata/test_labels.txt','r') as infile:
    testLabels = np.array(json.load(infile)).astype(np.float32)
infile.close()


# Create graph
sess = tf.Session()

x_vals = trainFeatures
y_vals = trainLabels

# declare variables
batch_size = 2048
num_features = 1152
num_labels = 10
regulation_rate = 5e1

x_data = tf.placeholder(shape=[None, 1152], dtype=tf.float32)
y_target = tf.placeholder(shape=[10, None], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, 1152], dtype=tf.float32)

b = tf.Variable(tf.random_normal(shape=[num_labels, batch_size]))
zeros = tf.zeros(shape=[num_labels, batch_size])

# linear kernel
my_kernel = tf.matmul(x_data, tf.transpose(x_data))

# Declare function to do reshape/batch multiplication
def reshape_matmul(mat):
    v1 = tf.expand_dims(mat, 1)
    v2 = tf.reshape(v1, [10, batch_size, 1])
    return(tf.matmul(v2, v1))

first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = reshape_matmul(y_target)

second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)),[1,2])
loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))
loss += regulation_rate * tf.nn.l2_loss(b)

# linear prediction kernel
pred_kernel = tf.matmul(x_data, tf.transpose(prediction_grid))

# calculate predictions
prediction_output = tf.matmul(tf.multiply(y_target,b), pred_kernel)
prediction = tf.arg_max(prediction_output, 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target,0)), tf.float32))


# define optimizer
my_opt = tf.train.AdamOptimizer(0.01)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
train_acc = []
valid_acc = []

# start running gradient descent
for i in range(2500):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = y_vals[:,rand_index]
    #rand_x = x_vals[:batch_size]
    #rand_y = y_vals[:,:batch_size]
    
    # feed data to optimizer batch by batch    
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    
    temp_train_acc =  sess.run(tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(trainLabels,0)), tf.float32)) ,feed_dict={x_data: rand_x, y_target: rand_y,prediction_grid: trainFeatures})
    train_acc.append(temp_train_acc)
    temp_valid_acc = sess.run(tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(validateLabels,0)), tf.float32)) ,feed_dict={x_data: rand_x, y_target: rand_y,prediction_grid: validateFeatures})
    valid_acc.append(temp_valid_acc)
    
    # print out accuracy every 50 steps
    if (i+1)%50==0:
        print('Step #' + str(i+1))
        print('Loss = ' + str(temp_loss))
        print('Mini Batch accuracy = ' + str(temp_train_acc))
        print('Validation accuracy = ' + str(temp_valid_acc))

acc_test = sess.run(tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(testLabels,1)), tf.float32)) ,feed_dict={x_data: rand_x, y_target: rand_y,prediction_grid: testFeatures})
print('Test accuracy = ' + str(acc_test))      

# Plot loss over time
num_steps = 2500
step = range(1, num_steps+1)
epoch = [i*batch_size/46171 for i in step]

plt.plot(epoch, loss_vec, 'k-')
plt.title('Linear Kernel SVM\nDual Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Dual Loss')
plt.savefig('./graphs/linearKernel_loss.png')
plt.show()
plt.close()
    
# Plot train and test accuracy
plt.plot(epoch, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(epoch, valid_acc, 'r--', label='Validate Set Accuracy')
plt.title('Linear Kernel SVM\nTrain and Validate Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('./graphs/linearKernel_acc.png')
plt.show()
plt.close()

