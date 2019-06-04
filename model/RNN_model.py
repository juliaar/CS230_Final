
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

    for k in range(len(filename)):

        # Read image in BGR format
        image = cv2.imread(filename[k])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB

        if k == 0:
            resized_image = image.flatten()
            print(resized_image)
            print(resized_image.shape)
        else:
            resized_image_k = image.flatten()
            resized_image = tf.stack([resized_image, resized_image_k])

    # outputs resizes image with shape (64,64,15) & its label
    return resized_image

def _input_def(filenames, labels, size):
    """Input function for the EgoGesture dataset.

    The filenames have format "{class label}_{part number (0-4)}_{total example number}.jpg".
    For instance: "data_dir/23_2_806.jpg".

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images (5 images together)
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    images = []
    for i in range(len(filenames)):
        im_i, _ = _parse_function(filenames[i], size)
        images.append(im_i)

    return images

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
    train_img = _input_def(train_filenames, train_labels, params.image_size)
    eval_img = _input_def(eval_filenames, eval_labels, params.image_size)

    # Parameters
    n_classes = params.num_labels
    learning_rate = params.learning_rate
    training_iters = params.training_iterations
    batch_size = params.batch_size
    n_layers = params.num_layers  # number of LSTM layers
    n_hidden = params.num_hidden  # number of hidden layers / cells in each LSTM layer

    # TODO: UPDATE
    n_input = params.num_input  # data is (img feature shape : 625 descriptors * 40 frames)
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

    # train_inputs, eval_inputs
    # inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}

    # Training Variables
    data = train_img
    print(data)
    label_y = train_labels
    print(label_y)
    data_x = []
    # TODO: update following
    one = 0
    two = 0
    three = 0
    four = 0

    # Testing Variables
    test_data = eval_img
    test_x = []
    test_label = eval_labels
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


