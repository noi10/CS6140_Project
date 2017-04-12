# Implementing Logistic Regression
#
# The the output of our model is the standard logistic regression:
# y = sigmoid(A * x + b)
# The x matrix input will have dimensions (batch size * features).  
# The y target output will have the dimension batch size * 10.
#
# The loss function we will use will be the mean of the cross-entropy loss:
# loss = mean( - y * log(predicted) + (1-y) * log(1-predicted) )
# TensorFlow has this cross entropy built in, and we can use the function, 
# tf.softmax_cross_entropy_with_logits()


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json

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

with open('./txtdata/yt8m_50_train_labels.txt','r') as infile:
    trainLabels = np.array(json.load(infile)).astype(np.float32)
infile.close()

with open('./txtdata/validate_labels.txt','r') as infile:
    validateLabels = np.array(json.load(infile)).astype(np.float32)
infile.close()

with open('./txtdata/test_labels.txt','r') as infile:
    testLabels = np.array(json.load(infile)).astype(np.float32)
    infile.close()

# Start a computational graph session.
graph = tf.Graph()
with graph.as_default():

    # declare variables
    batch_size = 256
    num_steps = 1500
    beta = 0.001
    base_learning_rate = 0.01
    learning_rate_decay_examples = 10000
    learning_rate_decay = 0.96
    num_labels = 10
    num_features = 1152

    # declare linear model
    def model(data):
      return tf.matmul(data, w_logit) + b_logit

    # accuracy function
    def accuracy(predictions, labels):
      return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])
    
    # calculate loss
    tf_train_dataset = tf.placeholder(shape=[batch_size, num_features], dtype=tf.float32 )
    tf_train_labels = tf.placeholder(shape=[batch_size, num_labels], dtype=tf.float32)
    tf_train_all_dataset = tf.constant(trainFeatures)
    tf_valid_dataset = tf.constant(validateFeatures)
    tf_test_dataset = tf.constant(testFeatures)

    w_logit = tf.Variable(tf.truncated_normal([num_features, num_labels]))
    b_logit = tf.Variable(tf.zeros([num_labels]))


    logits = model(tf_train_dataset)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    regularized_loss = tf.nn.l2_loss(w_logit)
    total_loss = loss + beta*regularized_loss

    # learning rate decay
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
      base_learning_rate,
      global_step,
      learning_rate_decay_examples,
      learning_rate_decay,
      staircase=True)
    
    # define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
    #optimizer = tf.train.AdamOptimizer(0.1).minimize(total_loss)

    # calculate predictions
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))
    all_train_prediction = tf.nn.softmax(model(tf_train_all_dataset))

    loss_vec = []
    train_acc = []
    valid_acc = []

    with tf.Session(graph=graph) as session:
      init = tf.global_variables_initializer()
      session.run(init)
      print("Initialize")
      
      # start running gradient descent
      for step in range(num_steps):
        rand_index = np.random.choice(len(trainFeatures), size=batch_size)
        batch_data = trainFeatures[rand_index]
        batch_labels = trainLabels[rand_index]

        # feed data to optimizer batch by batch
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        temp_loss = l
        loss_vec.append(temp_loss)
        temp_train_acc = accuracy(all_train_prediction.eval(), trainLabels)
        train_acc.append(temp_train_acc)
        temp_valid_acc = accuracy(valid_prediction.eval(), validateLabels)
        valid_acc.append(temp_valid_acc)
        
        # print out accuracy every 100 steps
        if (step%100==0):
          print("Minibatch loss at step %d: %f" % (step,l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Train accuracy: %.1f%%" % accuracy(all_train_prediction.eval(), trainLabels))
          print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), validateLabels))
      print("Test accuracy %.1f%%" % accuracy(test_prediction.eval(), testLabels))
    
    # Plot loss over time
    step = range(1, num_steps+1)
    epoch = [i*batch_size/46171 for i in step]
    
    plt.plot(epoch,loss_vec, 'k-')
    plt.title('Multinomial Logistic Regression\n Cross Entropy Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.savefig('./graphs/logistic_loss.png')
    plt.show()
    plt.close()
    
    # Plot train and test accuracy
    plt.plot(epoch, train_acc, 'k-', label='Train Set Accuracy')
    plt.plot(epoch, valid_acc, 'r--', label='Validate Set Accuracy')
    plt.title('Multinomial Logistic Regression \n Train and Validate Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('./graphs/logistic_acc.png')
    plt.show()
    plt.close()

