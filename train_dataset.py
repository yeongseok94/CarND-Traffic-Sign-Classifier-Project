#%%
import pickle
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

#%% Load the Data

training_file = 'dataset/train.p'
validation_file = 'dataset/valid.p'
testing_file = 'dataset/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

#%% Summary of Data Set

n_train = len(X_train)
n_valid = len(X_valid)
n_test = len(X_test)
image_shape = np.shape(X_train[0])
n_classes = len(np.unique(y_train))
print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#%% Data exploration visualization

plt.figure(figsize=(10, 10))
for i in range(0, 9):
    index = random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    ax = plt.subplot(3, 3, i+1)
    ax.imshow(image)
    ax.set_title('Sign Type = '+str(y_train[index]))
plt.savefig('fig1.png')

#%% Preprocessing

# Grayscale & Normalize Images
def grayscale_normalize(array):
    grayscale_data = []
    for i in range(0,array.shape[0]):
        grayscale_data.append(cv2.cvtColor(array[i], cv2.COLOR_RGB2GRAY))
    grayscale_data = np.array(grayscale_data)
    grayscale_data = grayscale_data.reshape([-1, image_shape[0], image_shape[1], 1])
    return (grayscale_data - 128.0) / 128.0

X_train = grayscale_normalize(X_train)
X_valid = grayscale_normalize(X_valid)
X_test = grayscale_normalize(X_test)

#%% Model Architecture

EPOCHS = 10
BATCH_SIZE = 64
rate = 0.001

x = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

# Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
mu = 0
sigma = 0.1

# Layer 1: Convolutional.
conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 32), mean = mu, stddev = sigma))
conv1_b = tf.Variable(tf.zeros(32))
conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
conv1 = tf.nn.relu(conv1)
conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# Layer 2: Convolutional.
conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), mean = mu, stddev = sigma))
conv2_b = tf.Variable(tf.zeros(64))
conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
conv2 = tf.nn.relu(conv2)
conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# Flatten.
fc0   = flatten(conv2)

# Layer 3: Fully Connected.
fc1_W = tf.Variable(tf.truncated_normal(shape=(1600, 320), mean = mu, stddev = sigma))
fc1_b = tf.Variable(tf.zeros(320))
fc1   = tf.matmul(fc0, fc1_W) + fc1_b
fc1    = tf.nn.relu(fc1)

# Layer 4: Fully Connected.
fc2_W  = tf.Variable(tf.truncated_normal(shape=(320, 43), mean = mu, stddev = sigma))
fc2_b  = tf.Variable(tf.zeros(43))
logits = tf.matmul(fc1, fc2_W) + fc2_b

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

#%% Function for validation and evaluation

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

#%% Model Training & Validation, Evaluation

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        train_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_valid, y_valid)
        test_accuracy = evaluate(X_test, y_test)
        print("EPOCH {} ...".format(i+1))
        print("Train Accuracy = {:.3f}".format(train_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Test Accuracy = {:.3f}".format(test_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")

#%% Load the new dataset & Preprocess

X_new = []
y_new = [0,1,2,3,4] # Prediction on the sign type
imgdir = 'dataset_new/'
imglist = ['20.jpg', '30.jpg', '50.jpg', '60.jpg', '70.jpg']
plt.figure(figsize=(15,3))
for i in range(0, 5):
    imgname = imglist[i]
    image = mpimg.imread(imgdir+imgname)
    plt.subplot(1, 5, i+1)
    plt.imshow(image)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    X_new.append(image)
X_new = np.array(X_new).reshape([-1, image_shape[0], image_shape[1], 1])
plt.savefig('fig2.png')


#%% Model Evaluation on new dataset & Top 5 Softmax Probabilities

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    test_accuracy = evaluate(X_new, y_new)
    print("Test Accuracy (NEW) = {:.3f}".format(test_accuracy))
    print()
    
    for i in range(0,5):
        X_current = X_new[i].reshape([-1, image_shape[0], image_shape[1], 1])
        tmp = sess.run(tf.nn.top_k(tf.nn.softmax(logits), k=5), feed_dict={x: X_current})
        
        values = list(tmp.values[0])
        indices = list(tmp.indices[0])
        maximum = max(values)
        rest = np.sum(values)-maximum
        print("Image%g-Probabilities:"%i, values , "Predicted Image:", indices)
        print("Max:%f"%maximum, "Others:%f"%rest) #, maximum+rest
    
        if i in indices:
            print("Image is in the top 5 predictions")
        else:
            print("Image is NOT in the top 5 predictions")
        print()
    