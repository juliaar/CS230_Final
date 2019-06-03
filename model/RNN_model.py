
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import argparse



def RNN_def(x, weights, biases, n_timesteps, n_hidden, n_layers, training):

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
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


def RNN_model(params, path):
    # path: path of data files

    # Parameters
    n_classes = params.num_labels
    learning_rate = params.learning_rate
    training_iters = params.training_iterations
    n_input = params.num_input  # data is (img feature shape : 625 descriptors * 40 frames)
    n_timesteps = params.num_timesteps  # timesteps = frames
    batch_size = params.batch_size
    n_layers = params.num_layers  # number of LSTM layers
    n_hidden = params.num_hidden  # number of hidden layers / cells in each LSTM layer

    # tf Graph input
    x = tf.placeholder("float", [None, n_timesteps, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Define weights & biases
    weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

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
    data = np.load('train_data.npy')
    label_y = []
    data_x = []
    # TODO: update following
    one = 0
    two = 0
    three = 0
    four = 0

    # Testing Variables
    test_data = np.load('test_data.npy')
    test_x = []
    test_label = []
    n_test = 120 # TODO: update as len(test_data)
    accuracy_counter = 0
    # TODO: update following
    One = 0
    Two = 0
    Three = 0
    Four = 0

    with tf.Session() as sess:
        sess.run(init)

        # Training Loop
        for i in range(training_iters):
            for n in range(batch_size):

                rand_n = np.random.random_integers(0, len(data) - 1)
                # print rand_n
                data_x.append(data[rand_n, :, :])

                if (0 <= rand_n <= 119):
                    label_y.append([1, 0, 0, 0])
                    one += 1

                elif (120 <= rand_n <= 239):
                    label_y.append([0, 1, 0, 0])
                    two += 1

                elif (240 <= rand_n <= 359):
                    label_y.append([0, 0, 1, 0])
                    three += 1

                elif (360 <= rand_n <= 479):
                    label_y.append([0, 0, 0, 1])
                    four += 1

            batch_x = np.array(data_x)

            batch_y = np.array(label_y)
            batch_x = batch_x.reshape((batch_size, n_timesteps, n_input))
            batch_y = batch_y.reshape((batch_size, n_classes))

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            if (i % 100) == 0:
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

        print(test_data.shape)
        # Testing Loop
        for i in range(n_test):
            test_x.append(test_data[i, :, :])

            if (0 <= i <= 29):
                label_y.append([1, 0, 0, 0])
                One += 1

            elif (30 <= i <= 59):
                label_y.append([0, 1, 0, 0])
                Two += 1

            elif (60 <= i <= 89):
                label_y.append([0, 0, 1, 0])
                Three += 1

            elif (90 <= i <= 119):
                label_y.append([0, 0, 0, 1])
                Four += 1

            batch_x = np.array(test_x)
            batch_y = np.array(label_y)
            batch_x = batch_x.reshape((1, n_timesteps, n_input))

            # Calculate batch accuracy
            acc = sess.run(accuracy_test, feed_dict={x: batch_x, y: batch_y})

            if acc != 0.0:
                accuracy_counter = accuracy_counter + 1

            print("Testing Accuracy:", acc)
            # print("The accuracy for testing per 4 iterations of each training sample is --  " +  "{:.5f}".format(a))

            print("Testing of class", i)

            del test_x[:]
            del label_y[:]

        print('Final accuracy = ', ((float(accuracy_counter)) / (float(n_test)) * float(100)), '%')


