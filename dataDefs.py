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
'''

import numpy as np
import os.path
import tensorflow as tf
import pandas as pd
import fnmatch
import re
import skimage
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


'''
For Julia to run in main.py:
from dataDefs import get_data
EgoGesture_path = 'D:\CS230-Datasets\EgoGesture'
gestures = [23, 52, 53]
percent_dev = 5
percent_test = 5
train_set, dev_set, test_set = get_data(EgoGesture_path, gestures, percent_dev, percent_test)
X_train, Y_train, C_train = train_set
print(X_train[0].shape) # print the shape of the first training example to check
'''

def find_data_and_labels(EgoGesture_path, gestures):
    '''
        Inputs:
    EgoGesture_path (path to the folder with all data)
    gestures (labels for gestures (n), number of classes is number of gestures + 1)
        Returns:
    yc (index of class in y (length n+1), yc[0] gives a vector y for the first example, all values zero but one),
    path_images (the path for the images, path_images[0][1] gives the second image of the first example)
    class_label (the label of the class (as numbered by EgoGesture), value between 1 and 83)
    '''
    n_gestures = len(gestures)
    yc = []
    class_label = []
    path_images = []
    path_images_start = os.path.join(EgoGesture_path, 'images_320-240')
    path_labels = os.path.join(EgoGesture_path, 'labels-final-revised1')
    l_subject_dirs = [x[0] for x in tf.gfile.Walk(path_labels)]
    for l_subject_dir in l_subject_dirs:
        # We only want the Scene# file paths, don't include others:
        if not ('Scene' in l_subject_dir):
            continue
        n_groups = len(fnmatch.filter(os.listdir(l_subject_dir), '*.csv'))
        group_name = "Group"
        group_name = [group_name + str(i+1) + '.csv' for i in range(n_groups)]
        for i in range(n_groups):
            path_group = os.path.join(l_subject_dir, group_name[i])
            data_i = pd.read_csv(path_group, header=None)
            n_gestures_i = len(data_i[0])
            for j in range(n_gestures_i):
                for cind in range(len(gestures)):
                    c = gestures[cind]
                    if data_i[0][j] == c:
                        y = np.zeros((n_gestures+1, 1))
                        y[cind] = 1
                        yc.append(y)
                        pics = create_pic_path(path_images_start, path_group, data_i, j)
                        path_images.append(pics)
                        class_label.append(gestures[cind])
    #
    return yc, path_images, class_label

def pick_5_images(data_i, j):
    c_start = data_i[1][j]
    c_end = data_i[2][j]
    total = c_end - c_start + 1
    ave = int(round((c_end+c_start)/2))
    if total/3 > 5:
        pics = [ave-6, ave-3, ave, ave+3, ave+6]
    elif total/2 > 5:
        pics = [ave-4, ave-2, ave, ave+2, ave+4]
    else:
        pics = [ave-2, ave-1, ave, ave+1, ave+2]
    #
    pic_files = []
    for i in range(len(pics)):
        pic_i = str(pics[i])
        for k in range(5):
            if len(pic_i) < 6:
                pic_i = '0' + pic_i
        pic_i = pic_i + '.jpg'
        pic_files.append(pic_i)
    #
    return pic_files

def create_pic_path(path_images_start, path_group, data_i, j):
    path_num = [float(s) for s in re.findall(r'-?\d+\.?\d*', path_group)]
    group_num = int(path_num[len(path_num)-1])
    scene_num = int(path_num[len(path_num) - 2])
    subject_num = int(path_num[len(path_num) - 3])
    if len(str(subject_num)) == 1:
        subject_num = '0' + str(subject_num)
    else:
        subject_num = str(subject_num)
    image_folder = os.path.join(path_images_start, 'Subject' + subject_num, 'Scene' + str(scene_num), 'Color', 'rgb' + str(group_num))
    #
    pic_files = pick_5_images(data_i, j)
    path_images = []
    for i in range(len(pic_files)):
        path_images.append(os.path.join(image_folder, pic_files[i]))
    #
    return path_images

def split_data_into_sets(m, percent_dev, percent_test):
    m_dev = int(round(m*percent_dev/100))
    m_test = int(round(m*percent_test/100))
    m_train = m - m_dev - m_test
    return m_train, m_dev, m_test

def get_data(EgoGesture_path, gestures, percent_dev, percent_test):
    '''
        Inputs:
    EgoGesture_path (path to the folder with all data)
    gestures        (labels for gestures (n), number of classes is number of gestures + 1)
    percent_dev, percent_test (those + percent_train = 100)
        Returns:
    train_set   ([X_train, Y_train, C_train])
    dev_set     ([X_dev, Y_dev, C_dev])
    test_set    ([X_test, Y_test, C_test])
                Where for each example:
                X is an input-array containing 5 stacked images
                Y is an vector of 0/1, with 1 in the correct class
                C is the class label
    '''
    yc, path_images, class_label = find_data_and_labels(EgoGesture_path, gestures)
    m_total = len(class_label)
    m_train, m_dev, m_test = split_data_into_sets(m_total, percent_dev, percent_test)
    #
    X_train = []
    Y_train = []
    C_train = []
    X_dev = []
    Y_dev = []
    C_dev = []
    X_test = []
    Y_test = []
    C_test = []
    #
    for i in range(0, m_total):
        if i < m_train:
            for pic_i in range(len(path_images[i])):
                #img_i = mpimg.imread(path_images[i][pic_i])/255
                img_i = skimage.io.imread(path_images[i][pic_i])/255
                if pic_i == 0:
                    img_stacked = img_i
                else:
                    img_stacked = np.dstack((img_stacked, img_i))
            X_train.append(img_stacked)
            Y_train.append(yc[i])
            C_train.append(class_label[i])
        elif i >= m_train and i < m_train+m_dev:
            for pic_i in range(len(path_images[i])):
                img_i = skimage.io.imread(path_images[i][pic_i])/255
                if pic_i == 0:
                    img_stacked = img_i
                else:
                    img_stacked = np.dstack((img_stacked, img_i))
            X_dev.append(img_stacked)
            Y_dev.append(yc[i])
            C_dev.append(class_label[i])
        else:
            for pic_i in range(len(path_images[i])):
                img_i = skimage.io.imread(path_images[i][pic_i])/255
                if pic_i == 0:
                    img_stacked = img_i
                else:
                    img_stacked = np.dstack((img_stacked, img_i))
            X_test.append(img_stacked)
            Y_test.append(yc[i])
            C_test.append(class_label[i])
    #
    train_set = [X_train, Y_train, C_train]
    dev_set = [X_dev, Y_dev, C_dev]
    test_set = [X_test, Y_test, C_test]
    return train_set, dev_set, test_set