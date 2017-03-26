import numpy as np
import tensorflow as tf
import json

graph = tf.Graph()
with graph.as_default():
    batch_size = 128
    beta = 0.001
    num_labels = 10

    with open('./txtdata/train_features.txt','r') as infile:
        trainFeatures = np.array(json.load(infile)).astype(np.float32)
    infile.close()

    with open('./txtdata/validate_features.txt','r') as infile:
        validateFeatures = np.array(json.load(infile)).astype(np.float32)
    infile.close()

    with open('./txtdata/test_features.txt','r') as infile:
        testFeatures = np.array(json.load(infile)).astype(np.float32)
    infile.close()

    with open('./txtdata/train_labels.txt','r') as infile:
        trainLabels = np.array(json.load(infile)).astype(np.float32)
    infile.close()

    with open('./txtdata/validate_labels.txt','r') as infile:
        validateLabels = np.array(json.load(infile)).astype(np.float32)
    infile.close()

    with open('./txtdata/test_labels.txt','r') as infile:
        testLabels = np.array(json.load(infile)).astype(np.float32)
    infile.close()


    def trainModel(data):
        Y1 = tf.nn.relu(tf.matmul(data, w1_logit) + b1_logit)
        Y1 = tf.nn.dropout(Y1, pkeep)
        Y2 = tf.nn.relu(tf.matmul(Y1, w2_logit) + b2_logit)
        Y2 = tf.nn.dropout(Y2, pkeep)
        return tf.matmul(Y2, w_logit) + b_logit

    def evalModel(data):
        Y1 = tf.nn.relu(tf.matmul(data, w1_logit) + b1_logit)
        Y1 = tf.nn.dropout(Y1, 1)
        Y2 = tf.nn.relu(tf.matmul(Y1, w2_logit) + b2_logit)
        Y2 = tf.nn.dropout(Y2, 1)
        return tf.matmul(Y2, w_logit) + b_logit

    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                /predictions.shape[0])

    num_labels = 10
    num_features = 1152
    K = 2304
    L = 1152
    pkeep = 0.75
    tf_train_dataset = tf.placeholder(shape=[batch_size, num_features], dtype=tf.float32 )
    tf_train_labels = tf.placeholder(shape=[batch_size, num_labels], dtype=tf.float32)
    tf_valid_dataset = tf.constant(validateFeatures)
    tf_test_dataset = tf.constant(testFeatures)

    w1_logit = tf.Variable(tf.truncated_normal([num_features, K]))
    b1_logit = tf.Variable(tf.zeros([K]))
    w2_logit = tf.Variable(tf.truncated_normal([K, L]))
    b2_logit = tf.Variable(tf.zeros([L]))

    w_logit = tf.Variable(tf.truncated_normal([L, num_labels]))
    b_logit = tf.Variable(tf.zeros([num_labels]))

    logits = trainModel(tf_train_dataset)    

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    regularized_loss = tf.nn.l2_loss(w_logit)
    total_loss = loss + beta*regularized_loss

    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(total_loss) 

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(evalModel(tf_valid_dataset))
    test_prediction = tf.nn.softmax(evalModel(tf_test_dataset))

    num_steps = 1000000

    with tf.Session(graph=graph) as session:
        init = tf.global_variables_initializer()
        session.run(init)
        print("Initialize")
        for step in range(num_steps):
            rand_index = np.random.choice(len(trainFeatures), size=batch_size)
            batch_data = trainFeatures[rand_index]
            batch_labels = trainLabels[rand_index]

            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

            if (step%500==0):
                print("Minibatch loss at step %d: %f" % (step,l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), validateLabels))
        print("Test accuracy %.1f%%" % accuracy(test_prediction.eval(), testLabels))



