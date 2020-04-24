#%%
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.contrib.layers import flatten
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#%%
imagelist = os.listdir("dataset_new/")