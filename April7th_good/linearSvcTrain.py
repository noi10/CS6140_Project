import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
from tensorflow.python.framework import ops

num_labels = 10
num_features = 1152
delta = 15.0
regulation_rate = 5e-4
batch_size = 256
num_steps = 50000
starter_learning_rate = 0.1

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

with open('./txtdata/validate_labels.txt','r') as infile:
    validateLabels = np.array(json.load(infile)).astype(np.float32)
infile.close()

with open('./txtdata/test_labels.txt','r') as infile:
    testLabels = np.array(json.load(infile)).astype(np.float32)
infile.close()
    

graph = tf.Graph()
with graph.as_default():

    def model(data):
        return tf.matmul(data, w_logit) + b_logit


    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                /predictions.shape[0])
    

    tf_train_dataset = tf.placeholder(shape=[batch_size, num_features], dtype=tf.float32 )
    tf_train_labels = tf.placeholder(shape=[batch_size, num_labels], dtype=tf.float32)
    #tf_train_dataset = tf.constant(trainFeatures)
    #tf_train_labels = tf.constant(trainLabels)
    tf_valid_dataset = tf.constant(validateFeatures)
    tf_test_dataset = tf.constant(testFeatures)

    
    w_logit = tf.Variable(tf.truncated_normal([num_features, num_labels]))
    b_logit = tf.Variable(tf.zeros([num_labels]))

    logits = model(tf_train_dataset)
    #logits = tf.matmul(tf_train_dataset, weights) + biases

    #y = tf.reduce_sum(logits * tf_train_labels, 1, keep_dims=True)

    #loss = tf.reduce_mean(tf.reduce_sum(tf.maximum(0.0, logits - y + delta), 1)) - delta
    loss = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(logits, tf_train_labels) ) ))
    loss += regulation_rate * tf.nn.l2_loss(w_logit)


    global_step = tf.Variable(0, trainable=False)
    

    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 250, 0.7, staircase=True)
    learning_rate = starter_learning_rate
    #optimizer = train_step(loss, generation_num)
    #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

    with tf.Session(graph = graph) as session:
        init = tf.global_variables_initializer()
        session.run(init)

        for step in range(num_steps):
            rand_index = np.random.choice(len(trainFeatures), size=batch_size)
            batch_data = trainFeatures[rand_index]
            batch_labels = trainLabels[rand_index]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)
            if (step%100==0):
                print("Minibatch loss at step %d: %f" % (step,l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                #print(session.run(predictions))
                print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), validateLabels))
        print("Test accuracy %.1f%%" % accuracy(test_prediction.eval(), testLabels))
