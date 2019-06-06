# Use Keras and LSTM to train the model

import time
import os.path
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten, Activation, BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling2D)
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical


def add_default_block(model, filters, init, reg_lambda):
    # conv
    model.add(TimeDistributed(Conv2D(filters, (3, 3), padding='same',
                                     kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(reg_lambda))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    # conv
    model.add(TimeDistributed(Conv2D(filters, (3, 3), padding='same',
                                     kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(reg_lambda))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    # max pool
    #model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    return model

def lrcn(input_shape, n_classes, n_cnnlayers, LSTM_keep_prop, LSTM_units):
    """ Build a CNN into RNN.
    Based on: https://github.com/harvitronix/five-video-classification-methods/blob/master/models.py
    """

    initialiser = 'glorot_uniform'
    reg_lambda = 0.001  # kernel_regularizer

    model = Sequential()

    # ~~ CONV2D ~~

    # first (non-default) double-layer block
    model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), padding='same',
                                     kernel_initializer=initialiser,
                                     kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
                              input_shape=input_shape))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(32, (3, 3), kernel_initializer=initialiser,
                                     kernel_regularizer=tf.keras.regularizers.l2(reg_lambda))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    # 2nd-5th (default) double-layer blocks
    layers = 2
    filters = 32
    while layers < n_cnnlayers:
        filters = filters*2
        model = add_default_block(model, filters, init=initialiser, reg_lambda=reg_lambda)
        layers = layers + 2

    # ~~ LSTM ~~

    # LSTM output head
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(LSTM_units, return_sequences=False, dropout=LSTM_keep_prop))
    model.add(Dense(n_classes, activation='softmax'))

    return model

def process_image(image_file, image_size):
    """Given an image, process it and return the array."""
    # Load the image.
    image = load_img(image_file, target_size=(image_size, image_size))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    x = (img_arr / 255.).astype(np.float32)

    return x

def convert_to_one_hot(num_classes, class_str):

    classes = np.arange(num_classes)
    classes = classes.tolist()

    # Encode it first.
    label_encoded = classes.index(class_str)

    # Now one-hot it.
    label_hot = to_categorical(label_encoded, num_classes)

    assert len(label_hot) == num_classes

    return label_hot

def convert_to_array(filenames, labels, batch_size, image_size, num_classes):
    while True:
        X, y = [], []
        for i in range(batch_size):
            sequence = []
            for file in filenames[i]:
                img = process_image(file, image_size)
                sequence.append(img)
            # Append:
            X.append(sequence)
            y.append(convert_to_one_hot(num_classes, labels[i]))
        yield np.array(X), np.array(y)

def train_LRCN_model(params, train_filenames, train_labels, eval_filenames, eval_labels):

    # Parameters
    learning_rate = params.learning_rate
    learning_decay = params.decay  # 1e-6
    n_classes = params.num_labels
    n_epochs = params.num_epochs
    batch_size = params.batch_size
    img_size = params.image_size
    n_timesteps = params.num_timesteps  # timesteps = frames
    early_stop = params.early_stop
    n_cnnlayers = params.num_cnn_layers  # 2, 4, 6, 8, or 10 only (otherwise just to big for how this is built)
    LSTM_keep_prop = params.LSTM_keep_prop
    LSTM_units = params.LSTM_units

    # Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('lrcn_data', 'checkpoints', 'lrcn-images.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # TensorBoard
    tb = TensorBoard(log_dir=os.path.join('lrcn_data', 'logs', 'lrcn'))

    # Stop when we stop learning.
    early_stopper = EarlyStopping(patience=early_stop)
    # patience = number of epochs before stopping once the loss starts to increase or stops improving
    # make it higher for higher learning rate (more zig-zag) and vice versa

    # Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('lrcn_data', 'logs', 'lrcn-training-' + str(timestamp) + '.log'))


    # Get generators.
    generator = convert_to_array(train_filenames, train_labels, batch_size, img_size, n_classes)
    val_generator = convert_to_array(eval_filenames, eval_labels, batch_size, img_size, n_classes)

    # Get the CNN-LSTM model.
    input_shape = (n_timesteps, img_size, img_size, 3)
    lrcn_model = lrcn(input_shape, n_classes, n_cnnlayers, LSTM_keep_prop, LSTM_units)

    # Compile the network.
    optimizer = Adam(lr=learning_rate, decay=learning_decay)
    lrcn_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print(lrcn_model.summary())

    # Get samples per epoch.
    steps_per_epoch = len(train_labels) // batch_size  # Would be automatically this anyway
    val_steps = len(eval_labels) // batch_size

    print("~~~~ going on to fitting!! ~~~~")
    # Fit! Use fit generator.
    lrcn_model.model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs,
        verbose=1,
        callbacks=[tb, early_stopper, csv_logger, checkpointer],
        validation_data=val_generator,
        validation_steps=val_steps,
        workers=1,
        use_multiprocessing=False)
    # where verbose mode: 0 = silent, 1 = progress bar, 2 = one line per epoch

