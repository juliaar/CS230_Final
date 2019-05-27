import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

###

def compute_cost(fname):
    #fname = "images/thumbs_up.jpg"
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(64,64))
    plt.imshow(my_image)

    return ??

###
