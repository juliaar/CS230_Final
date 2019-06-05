
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import argparse
import cv2
import tensorflow as tf
import numpy as np

def _parse_function(filename):
    """ Obtain the image from the filename. """

    resized_image_k = []

    for k in range(len(filename)):

        # Read image in BGR format
        image = cv2.imread(filename[k])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB

        resized_image_k.append(image.flatten())

    resized_image = tf.stack([resized_image_k[0], resized_image_k[1], resized_image_k[2], resized_image_k[3], resized_image_k[4],])

    return resized_image

def _input_def(filenames, labels):
    """Input function for the EgoGesture dataset.

    The filenames have format "{class label}_{part number (0-4)}_{total example number}.jpg".
    For instance: "data_dir/23_2_806.jpg".

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images (5 images together)
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.image_size`)
    """

    assert len(filenames) == len(labels), "Filenames and labels should have same length"
    images = []
    for i in range(len(filenames)):
        im_i = _parse_function(filenames[i])
        images.append(im_i)

    return images

def convert_to_one_hot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

def RNN_def(x, weights, biases, n_timesteps, n_hidden, n_layers, training):

    # Unstack to get a list of 'n_timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_timesteps, 1)

    # Defining a rnn-lstm cell with tensorflow
    if training:
        multi_lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden, forget_bias=0.0) for _ in range(n_layers)])
    else:
        multi_lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden, forget_bias=0.0, reuse=True) for _ in range(n_layers)])

    # Get lstm cell output
    outputs, states = rnn.static_rnn(multi_lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def RNN_model(params, train_filenames, train_labels, eval_filenames, eval_labels):

    print("Decoding images ...")
    train_img = _input_def(train_filenames, train_labels)
    eval_img = _input_def(eval_filenames, eval_labels)

    # Parameters
    n_classes = params.num_labels
    learning_rate = params.learning_rate
    training_iters = params.training_iterations
    batch_size = params.batch_size
    n_layers = params.num_layers  # number of LSTM layers
    n_hidden = params.num_hidden  # number of hidden layers / cells in each LSTM layer

    n_input = 3 * params.image_size * params.image_size
    n_timesteps = params.num_timesteps  # timesteps = frames

    # tf Graph input
    x = tf.placeholder("float", [None, n_timesteps, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Define weights & biases
    weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

    print("Starting to run RNN ...")
    prediction = RNN_def(x, weights, biases, n_timesteps, n_hidden, n_layers, True)

    # Defining loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    test_prediction = RNN_def(x, weights, biases, n_timesteps, n_hidden, n_layers, False)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    correct_test = tf.equal(tf.argmax(test_prediction, 1), tf.argmax(y, 1))
    accuracy_test = tf.reduce_mean(tf.cast(correct_test, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Training Variables
    data = train_img
    data_labels = convert_to_one_hot(np.array(train_labels), n_classes)
    data_x = []
    label_y = []

    # Testing Variables
    test_data = eval_img
    test_labels = convert_to_one_hot(np.array(eval_labels), n_classes)
    test_x = []
    n_test = len(eval_labels)
    accuracy_counter = 0

    print("Starting to train ...")
    with tf.Session() as sess:
        sess.run(init)

        # Training Loop
        for i in range(training_iters):
            for n in range(batch_size):

                rand_n = np.random.random_integers(0, len(data) - 1)
                tensor_n = data[rand_n]
                array_n = tensor_n.eval()
                data_x.append(array_n)
                label_y.append(data_labels[rand_n, :])

            batch_x = np.array(data_x)
            batch_y = np.array(label_y)

            batch_x = batch_x.reshape((batch_size, n_timesteps, n_input))
            batch_y = batch_y.reshape((batch_size, n_classes))

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            if (i % 50) == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})

                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iteration " + str(i) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
                print("~~~~~~~~")
            del data_x[:]
            del label_y[:]

        # Testing Loop
        for i in range(n_test):
            tensor_i = test_data[i]
            array_i = tensor_i.eval()
            test_x.append(array_i)

            batch_x = np.array(test_x)
            batch_y = np.array(test_labels)
            batch_x = batch_x.reshape((1, n_timesteps, n_input))

            # Calculate batch accuracy
            acc = sess.run(accuracy_test, feed_dict={x: batch_x, y: batch_y})

            if acc != 0.0:
                accuracy_counter = accuracy_counter + 1

            #print("Testing Accuracy:", acc)
            #print("Testing of sample", i)

            del test_x[:]
            del label_y[:]

        print('Final accuracy = ', ((float(accuracy_counter)) / (float(n_test)) * float(100)), '%')


