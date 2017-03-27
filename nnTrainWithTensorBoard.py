import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json

graph = tf.Graph()
with graph.as_default():
    batch_size = 256
    beta = 0.001
    num_steps = 500
    base_learning_rate = 0.003
    learning_rate_decay_examples = 360
    learning_rate_decay = 0.96
    num_labels = 10
    num_features = 1152
    K = 300
    L = 150
    M = 60
    N = 30
    #pkeep = 0.9
    
    with open('./txtdata/yt8m_100/train_features.txt','r') as infile:
        trainFeatures = np.array(json.load(infile)).astype(np.float32)
    infile.close()

    with open('./txtdata/yt8m_100/validate_features.txt','r') as infile:
        validateFeatures = np.array(json.load(infile)).astype(np.float32)
    infile.close()

    with open('./txtdata/yt8m_100/test_features.txt','r') as infile:
        testFeatures = np.array(json.load(infile)).astype(np.float32)
    infile.close()

    with open('./txtdata/yt8m_100/train_labels.txt','r') as infile:
        trainLabels = np.array(json.load(infile)).astype(np.float32)
    infile.close()

    with open('./txtdata/yt8m_100/validate_labels.txt','r') as infile:
        validateLabels = np.array(json.load(infile)).astype(np.float32)
    infile.close()

    with open('./txtdata/yt8m_100/test_labels.txt','r') as infile:
        testLabels = np.array(json.load(infile)).astype(np.float32)
    infile.close()

    def variable_summaries(var):
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

    def model(data, train=True):
      if train == False:
        pkeep = 1
      with tf.name_scope("relu_1"):       
        with tf.name_scope('Wx_plus_b'):
          preactivate1 = tf.matmul(data, w1_logit) + b1_logit
          tf.summary.histogram('pre_activations', preactivate1)
        activations1 = tf.nn.relu(preactivate1, name='activation')
        tf.summary.histogram('activations', activations1)
        # activations1 = tf.nn.dropout(Y1, pkeep)
      
      with tf.name_scope("relu_2"):        
        with tf.name_scope('Wx_plus_b'):
          preactivate2 = tf.matmul(activations1, w2_logit) + b2_logit
          tf.summary.histogram('pre_activations', preactivate2)
        activations2 = tf.nn.relu(preactivate2, name='activation')
        tf.summary.histogram('activations', activations2)
        # activations2 = tf.nn.dropout(Y2, pkeep)
      
      with tf.name_scope("relu_3"): 
        with tf.name_scope('Wx_plus_b'):
          preactivate3 = tf.matmul(activations2, w3_logit) + b3_logit
          tf.summary.histogram('pre_activations', preactivate3)
        activations3 = tf.nn.relu(preactivate3, name='activation')
        tf.summary.histogram('activations', activations3)
        # activations3 = tf.nn.dropout(Y3, pkeep)
      
      with tf.name_scope("relu_4"):        
        with tf.name_scope('Wx_plus_b'):
          preactivate4 = tf.matmul(activations3, w4_logit) + b4_logit
          tf.summary.histogram('pre_activations', preactivate4)
        activations4 = tf.nn.relu(preactivate4, name='activation')
        tf.summary.histogram('activations', activations4)
        # activations4 = tf.nn.dropout(Y4, pkeep)
      
      with tf.name_scope("softmax"):
        with tf.name_scope('Wx_plus_b'):
          logits = tf.matmul(activations4, w_logit) + b_logit
          tf.summary.histogram('logits', logits)
      return logits

    def accuracy(predictions, labels):
        acc = (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])
        return acc

    
    tf_train_dataset = tf.placeholder(shape=[batch_size, num_features], dtype=tf.float32 )
    tf_train_labels = tf.placeholder(shape=[batch_size, num_labels], dtype=tf.float32)
    tf_train_all_dataset = tf.constant(trainFeatures)
    tf_valid_dataset = tf.constant(validateFeatures)
    tf_test_dataset = tf.constant(testFeatures)
    
    with tf.name_scope("relu_1"):
      with tf.name_scope('weights'):
        w1_logit = tf.Variable(tf.truncated_normal([num_features, K], stddev = 0.1))
        variable_summaries(w1_logit)
      with tf.name_scope('biases'):
        b1_logit = tf.Variable(tf.zeros([K]))
        variable_summaries(b1_logit)
    with tf.name_scope("relu_2"):
      with tf.name_scope('weights'):
        w2_logit = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
        variable_summaries(w2_logit)
      with tf.name_scope('biases'):
        b2_logit = tf.Variable(tf.zeros([L]))
        variable_summaries(b2_logit)
    with tf.name_scope("relu_3"):
      with tf.name_scope('weights'):
        w3_logit = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
        variable_summaries(w3_logit)
      with tf.name_scope('biases'):
        b3_logit = tf.Variable(tf.zeros([M]))
        variable_summaries(b3_logit)
    with tf.name_scope("relu_4"):
      with tf.name_scope('weights'):
        w4_logit = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
        variable_summaries(w4_logit)
      with tf.name_scope('biases'):
        b4_logit = tf.Variable(tf.zeros([N]))
        variable_summaries(b4_logit)
    with tf.name_scope("softmax"):
      with tf.name_scope('weights'):
        w_logit = tf.Variable(tf.truncated_normal([N, num_labels]))
        variable_summaries(w_logit)
      with tf.name_scope('biases'):
        b_logit = tf.Variable(tf.zeros([num_labels]))
        variable_summaries(b_logit)

    logits = model(tf_train_dataset)    
    
    with tf.name_scope('loss'):
      with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)
        cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)
      with tf.name_scope('total_loss'):
        regularized_loss = tf.nn.l2_loss(w_logit)
        total_loss = cross_entropy + beta*regularized_loss
        tf.summary.scalar('total_loss', total_loss)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
      base_learning_rate,
      global_step,
      learning_rate_decay_examples,
      learning_rate_decay,
      staircase=True)
    with tf.name_scope('train'):
      optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step) 

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset, False))
    test_prediction = tf.nn.softmax(model(tf_test_dataset, False))
    all_train_prediction = tf.nn.softmax(model(tf_train_all_dataset, False))

    loss_vec = []
    train_acc = []
    valid_acc = []
    
    with tf.Session(graph=graph) as sess:
        
        # Merge all the summaries and write them out to ./nn_logs
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./nn_logs', sess.graph)
        tf.global_variables_initializer().run()        

        print("Initialize")
        for step in range(num_steps):
            rand_index = np.random.choice(len(trainFeatures), size=batch_size)
            batch_data = trainFeatures[rand_index]
            batch_labels = trainLabels[rand_index]

            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, loss, predictions, summary = sess.run([optimizer, cross_entropy, train_prediction, merged], feed_dict=feed_dict)

            loss_vec.append(loss)
            batch_train_acc = accuracy(predictions, batch_labels)
            temp_train_acc = accuracy(all_train_prediction.eval(), trainLabels)
            train_acc.append(temp_train_acc)
            temp_valid_acc = accuracy(valid_prediction.eval(), validateLabels)
            valid_acc.append(temp_valid_acc)
            
            train_writer.add_summary(summary, step)            

            if (step%50==0):
                print("Minibatch loss at step %d: %f" % (step,loss))
                print("Minibatch accuracy: %.1f%%" % batch_train_acc)
                print("Train accuracy: %.1f%%" % temp_train_acc)
                print("Validation accuracy: %.1f%%" % temp_valid_acc)
        
        print("Test accuracy %.1f%%" % accuracy(test_prediction.eval(), testLabels))

    # Plot loss over time
    plt.plot(loss_vec, 'k-')
    plt.title('Cross Entropy Loss per Step')
    plt.xlabel('Step')
    plt.ylabel('Cross Entropy Loss')
    plt.savefig('./nn_logs/nn_loss.png')
    plt.close()
    
    # Plot train and test accuracy
    plt.plot(train_acc, 'k-', label='Train Set Accuracy')
    plt.plot(valid_acc, 'r--', label='Validate Set Accuracy')
    plt.title('Train and Validate Accuracy')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('./nn_logs/nn_acc.png')
    plt.close()

