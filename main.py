'''
Have to be installed via terminal:
    pip install --upgrade tensorflow
    pip install --upgrade numpy
    pip install --upgrade h5py
    pip install --upgrade scipy
    pip install Pillow
    pip install matplotlib
    pip install pandas
    pip install image
    pip install scikit-image --upgrade
    pip install tqdm
    pip install keras
    pip install tabulate
'''

# COMMANDS:
# python train.py --data_dir "C:\Users\Julia Arnardottir\PycharmProjects\VisionExample\data\64x64_SIGNS" --model_dir "C:\Users\Julia Arnardottir\PycharmProjects\VisionExample\experiments\base_model"
# python search_hyperparams.py --data_dir "C:\Users\Julia Arnardottir\PycharmProjects\VisionExample\data\64x64_SIGNS" --parent_dir "C:\Users\Julia Arnardottir\PycharmProjects\VisionExample\experiments\learning_rate"
# python synthesize_results.py --parent_dir "C:\Users\Julia Arnardottir\PycharmProjects\VisionExample\experiments\learning_rate"
# python evaluate.py --data_dir "C:\Users\Julia Arnardottir\PycharmProjects\VisionExample\data\64x64_SIGNS" --model_dir "C:\Users\Julia Arnardottir\PycharmProjects\VisionExample\experiments\base_model"

'''
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from model.utilities import splitter

from scipy.special import softmax
import os.path
from dataDefs import get_data
import pandas as pd
import fnmatch

print(splitter)
'''

#import AdditionalDefs
#from convDefs import create_placeholders, initialize_parameters, forward_propagation
''' Summary of functions:
    zero_pad                    random_mini_batches
    conv_single_step            conv_forward
    pool_forward                conv_backward
    create_mask_from_window     distribute_value 
    pool_backward               create_placeholders
    initialize_parameters       forward_propagation
    compute_cost                model
'''

'''
print("skuli minn")

# Classes: [0,1,2,3] = [frame, applaud, heart, other] = [F, A, H, O]

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)


X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T


# HYPERPARAMETERS
num_epochs = 100
minibatch_size = 64
learning_rate = 0.009

model(X_train, Y_train, X_test, Y_test, learning_rate, num_epochs, minibatch_size, print_cost=True)
'''

'''
EgoGesture_path = 'D:\CS230-Datasets\EgoGesture'
gestures = [23, 52, 53]
percent_dev = 5
percent_test = 5
train_set, dev_set, test_set = get_data(EgoGesture_path, gestures, percent_dev, percent_test)
X_train, Y_train, C_train = train_set
print(X_train[0].shape)
#print(yc[3][0])

#tdt_folders = ['train_gestures', 'dev_gestures', 'test_gestures']
#print(tdt_folders[1])

import os
from tqdm import tqdm

#for i in tqdm(range(len(dev_filenames))):
#    print(dev_filenames[i])

data_dir = 'D:\\CS230-Datasets\\EgoGesture\\64x64_gestures'
dev_data_dir = os.path.join(data_dir, "dev_gestures")
eval_filenames = [os.path.join(dev_data_dir, f) for f in os.listdir(dev_data_dir)
                  if f.endswith('.jpg')]
eval_labels = [int(f.split('\\')[-1][:2]) for f in eval_filenames]

print(eval_filenames)
print(eval_labels)

'''


import tensorflow as tf

'''
listinn = [['a','b'],['c']]

listinn[0].append('k')
print(listinn)
#print(len(listinn[0]))

listi2 = []
listi2.append([1,2,3])
listi2.append([4,5,6])
#print(listi2)
#print(listi2[1][:])

for l in listinn:
    print(l)
'''
'''
w = 64
h = 64
x1 = tf.placeholder(shape=[None, w, h, 3], dtype='float32')
x2 = tf.placeholder(shape=[None, w, h, 3], dtype='float32')
x3 = tf.placeholder(shape=[None, w, h, 3], dtype='float32')
x4 = tf.placeholder(shape=[None, w, h, 3], dtype='float32')
x5 = tf.placeholder(shape=[None, w, h, 3], dtype='float32')
x2 = tf.concat([x1, x2], axis=3)
x2 = tf.concat([x2, x3, x4, x5], axis=3)
print(tf.shape(x2))
print(x2.shape)
#[None, x, h, 15]

#with tf.Session() as sess:
#    sess.run([y_hat, loss], feed_dict={x1: image1, x2: image2…, x5: image5})
'''
'''

l1 = ['abc', 'def', 'ghi']
l2 = tf.convert_to_tensor(l1)
print(l2)
print(l2.shape)
l3 = tf.reshape(l2, [3, 1])
print(l3)
print(l3.shape)
h, w = l3.shape
print(h)
print(w)

l4 = tf.Variable(l3)
print(l4)
print(l4.shape)
print(l4[0])

'''
w = 64
h = 64
x1 = tf.placeholder(shape=[None, w, h, 3], dtype='float32')
x2 = tf.placeholder(shape=[None, w, h, 3], dtype='float32')
x3 = tf.placeholder(shape=[None, w, h, 3], dtype='float32')
x4 = tf.placeholder(shape=[None, w, h, 3], dtype='float32')
x5 = tf.placeholder(shape=[None, w, h, 3], dtype='float32')
x2 = tf.concat([x1, x2], axis=3)
x2 = tf.concat([x2, x3, x4, x5], axis=3)
print(tf.shape(x2))
print(x2.shape)
im1, im2, im3, im4, im5 = tf.split(x2, num_or_size_splits=5, axis=3)
print(im3.shape)
#images = ['mynd1','mynd2','mynd3','mynd4','mynd5']
#im1, im2, im3, im4, im5 = tf.split(images, num_or_size_splits=5, axis=1)
#print(im3)
