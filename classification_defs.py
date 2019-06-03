import cv2
import os
from keras import backend as K
import numpy as np
from PIL import Image
import tensorflow as tf
from model.model_setup import build_model

# Stack and Classify

def _stackimages(files, params):
    # Run through the frames:
    for i in range(len(files)):
        image_string = tf.read_file(files[i])
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image_decoded, tf.float32)
        # Stack
        if i == 0:
            resized_image = tf.image.resize_images(image, [params.image_size, params.image_size])
            stacked_images = resized_image
        else:
            re_image_i = tf.image.resize_images(image, [params.image_size, params.image_size])
            stacked_images = tf.concat([stacked_images, re_image_i], axis=2)
    return stacked_images

def sample_def(files, params):
    '''
    files:  contains the files of the sample
    params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
    '''

    stacked_images = _stackimages(files, params)
    labels = K.placeholder(shape=(None, 1))
    stacked_images = stacked_images[None, :, :, :]
    sample_data = tf.data.Dataset.from_tensor_slices((stacked_images, labels)).batch(1).prefetch(1)
    iterator = sample_data.make_initializable_iterator()
    images, labels = iterator.get_next()
    input = {'images': images}
    print("images after dataset")
    print(images)

    with tf.variable_scope('model', reuse=False):
        # Compute the output distribution of the model and the predictions
        logit = build_model(False, input, params)
        prediction = tf.argmax(logit, 1)

    print("got to the end!")
    return prediction


