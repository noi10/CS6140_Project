import numpy as np
import tensorflow as tf
import json
def logisticTrain(batch, iterations):
  graph = tf.Graph()
  with graph.as_default():
    batch_size = batch
    beta = 0.001
    base_learning_rate = 0.01
    learning_rate_decay_examples = 10000
    learning_rate_decay = 0.96
    num_labels = 10
    num_features = 1152

    with open('./txtdata/yt8m_25/train_features.txt','r') as infile:
      trainFeatures = np.array(json.load(infile)).astype(np.float32)
    infile.close()

    with open('./txtdata/yt8m_25/validate_features.txt','r') as infile:
      validateFeatures = np.array(json.load(infile)).astype(np.float32)
    infile.close()

    with open('./txtdata/yt8m_25/test_features.txt','r') as infile:
      testFeatures = np.array(json.load(infile)).astype(np.float32)
    infile.close()

    with open('./txtdata/yt8m_25/train_labels.txt','r') as infile:
      trainLabels = np.array(json.load(infile)).astype(np.float32)
    infile.close()

    with open('./txtdata/yt8m_25/validate_labels.txt','r') as infile:
      validateLabels = np.array(json.load(infile)).astype(np.float32)
    infile.close()

    with open('./txtdata/yt8m_25/test_labels.txt','r') as infile:
      testLabels = np.array(json.load(infile)).astype(np.float32)
    infile.close()

    def model(data):
      return tf.matmul(data, w_logit) + b_logit

    def accuracy(predictions, labels):
      return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])
    
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

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
      base_learning_rate,
      global_step,
      learning_rate_decay_examples,
      learning_rate_decay,
      staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))
    all_train_prediction = tf.nn.softmax(model(tf_train_all_dataset))
    num_steps = iterations

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
          print("Train accuracy: %.1f%%" % accuracy(all_train_prediction.eval(), trainLabels))
          print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), validateLabels))
      print("Test accuracy %.1f%%" % accuracy(test_prediction.eval(), testLabels))

logisticTrain(128, 100000)
