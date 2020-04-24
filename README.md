# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./fig1.png "Exploartory"
[image2]: ./fig2.png "Traffic Signs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/yeongseok94/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python and numpy to calculate summary statistics of the traffic signs data set.

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32x32x3 RGB.
* The number of unique classes/labels in the data set is 43.

As a complementary material, this is what my code prints out.

```
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. This shows 9 random image from the training dataset and its corresponding label of sign types.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because reducing the input data size leads to fast and efficient classification of the given dataset, and grayscaling the image is just enough with preserving original image's features. Here, I used `cv2.cvtColor()` function to convert the given RGB image into grayscale.

As a second step, I normalized the image data because scaling data as zero-mean with low variance is generally good to train the network since we initialize the weights as zero-mean Gaussian with low variance and biases as zero. The normalizing analogy of pixel values is simply `(pixel-128)/128`.

As a last step, I augmented all the preprocessed images and reshaped the image data to have the dimension of `(# of images, 32, 32, 1)`, in order to make the dataset compatible with the rest of the code.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	                   					| 
|:---------------------:|:-----------------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x32        	|
| RELU					| -								        				|
| Max pooling	      	| 2x2 stride, outputs 14x14x32 		            		|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x64            |
| RELU					| -								        				|
| Max pooling	      	| 2x2 stride, outputs 5x5x64 		            		|
| Flatten       		| outputs 1600   						        		|
| Fully connected		| outputs 320  							        		|
| RELU          		| -           							        		|
| Fully connected		| outputs 43  							        		|
| Softmax          		| -           							        		|
| Output          		| 43 Probabilities						        		|
 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* Number of epochs: 10
* Batch size: 64
* Learning rate: 0.001
* Optimizer: `tf.compat.v1.train.AdamOptimizer()`

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy of 1.000
* Validation set accuracy of 0.950
* Test set accuracy of 0.942

As a complementary material, this is what my code prints out when training is ongoing.

```
Training...

EPOCH 1 ...
Train Accuracy = 0.964
Validation Accuracy = 0.885
Test Accuracy = 0.870

EPOCH 2 ...
Train Accuracy = 0.992
Validation Accuracy = 0.925
Test Accuracy = 0.904

EPOCH 3 ...
Train Accuracy = 0.993
Validation Accuracy = 0.919
Test Accuracy = 0.914

EPOCH 4 ...
Train Accuracy = 0.996
Validation Accuracy = 0.938
Test Accuracy = 0.922

EPOCH 5 ...
Train Accuracy = 0.995
Validation Accuracy = 0.935
Test Accuracy = 0.925

EPOCH 6 ...
Train Accuracy = 0.997
Validation Accuracy = 0.932
Test Accuracy = 0.932

EPOCH 7 ...
Train Accuracy = 0.998
Validation Accuracy = 0.943
Test Accuracy = 0.930

EPOCH 8 ...
Train Accuracy = 0.995
Validation Accuracy = 0.929
Test Accuracy = 0.919

EPOCH 9 ...
Train Accuracy = 0.999
Validation Accuracy = 0.947
Test Accuracy = 0.933

EPOCH 10 ...
Train Accuracy = 1.000
Validation Accuracy = 0.950
Test Accuracy = 0.942
```

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    * The first trial was from the LeNet with only the input and output size adapted to my dataset. This is because the LeNet is one of the most well-known basic convolutional neural network.
* What were some problems with the initial architecture?
    * The inital number of labels of the convolutional layers were only 6 and 16, where number of output labels is 43.
    * Since the size of image is not much big, 3 fully connected layer is not necessary.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    * The number of labels of the convolutional layers were tuned to 32 and 64, which can cover all the features of 43 labels.
    * The number of fully connected layer was reduced to 2. Then, the size of the second fully connected layer was decided to have the size of 320. The idea of deciding the size of this layer comes from the last layer the original LeNet which has about 8 times bigger size than the output layer.
* Which parameters were tuned? How were they adjusted and why?
    * By trial and error, the batch size was tuned to 64, which is quite small, since the image is small.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    * The number of labels of the convolutional layers should be large enough to cover all the number of features of outputs. However, this should be just a little bit bigger than the number of output labels. Excessively big number of labels of the convolutional layer just leads to computational burden and has no effect on the accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five traffic signs that I found on the web:

![alt text][image2]

The images might be difficult to classify because this is not actually German ones. These signs are Korean ones which have similar shapes but different fonts and thicker outer red rim.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are my code's printout:

```
Test Accuracy (NEW) = 0.800

Image0-Probabilities: [1.0, 0.0, 0.0, 0.0, 0.0] Predicted Image: [0, 1, 2, 3, 4]
Max:1.000000 Others:0.000000
Image is in the top 5 predictions

Image1-Probabilities: [1.0, 0.0, 0.0, 0.0, 0.0] Predicted Image: [1, 0, 2, 3, 4]
Max:1.000000 Others:0.000000
Image is in the top 5 predictions

Image2-Probabilities: [1.0, 0.0, 0.0, 0.0, 0.0] Predicted Image: [2, 0, 1, 3, 4]
Max:1.000000 Others:0.000000
Image is in the top 5 predictions

Image3-Probabilities: [1.0, 0.0, 0.0, 0.0, 0.0] Predicted Image: [2, 0, 1, 3, 4]
Max:1.000000 Others:0.000000
Image is in the top 5 predictions

Image4-Probabilities: [1.0, 0.0, 0.0, 0.0, 0.0] Predicted Image: [4, 0, 1, 2, 3]
Max:1.000000 Others:0.000000
Image is in the top 5 predictions
```

Here are the results of the prediction:

| Image			        |     Prediction	    | 
|:---------------------:|:---------------------:| 
| Speed limit (20km/h)  | Speed limit (20km/h)	|
| Speed limit (30km/h)  | Speed limit (30km/h)	|
| Speed limit (50km/h)  | Speed limit (50km/h)	|
| Speed limit (60km/h)  | Speed limit (50km/h)	|
| Speed limit (70km/h)  | Speed limit (70km/h)	|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of the previous ones, which gives an accuracy of 94.2%. Let's talk about why the accuracy is lower in the new sets.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is very sure that this is a speed limit sign of 20km/h, and the network checks it correctly. The top five soft max probabilities were:

| Probability |     Prediction	      | 
|:-----------:|:---------------------:| 
| 1.0         | Speed limit (20km/h)  |
| 0.0         | Speed limit (30km/h)  |
| 0.0         | Speed limit (50km/h)  |
| 0.0         | Speed limit (60km/h)  |
| 0.0         | Speed limit (70km/h)  |

For the second image, the model is very sure that this is a speed limit sign of 30km/h, and the network checks it correctly. The top five soft max probabilities were:

| Probability |     Prediction	      | 
|:-----------:|:---------------------:| 
| 1.0         | Speed limit (30km/h)  |
| 0.0         | Speed limit (20km/h)  |
| 0.0         | Speed limit (50km/h)  |
| 0.0         | Speed limit (60km/h)  |
| 0.0         | Speed limit (70km/h)  |

For the first image, the model is very sure that this is a speed limit sign of 50km/h, and the network checks it correctly. The top five soft max probabilities were:

| Probability |     Prediction	      | 
|:-----------:|:---------------------:| 
| 1.0         | Speed limit (50km/h)  |
| 0.0         | Speed limit (20km/h)  |
| 0.0         | Speed limit (30km/h)  |
| 0.0         | Speed limit (60km/h)  |
| 0.0         | Speed limit (70km/h)  |

For the first image, the model is very sure that this is a speed limit sign of 60km/h, but the network has confident in that the sign is surely a speed limit sign of 50km/h. The top five soft max probabilities were:

| Probability |     Prediction	      | 
|:-----------:|:---------------------:| 
| 1.0         | Speed limit (50km/h)  |
| 0.0         | Speed limit (20km/h)  |
| 0.0         | Speed limit (30km/h)  |
| 0.0         | Speed limit (60km/h)  |
| 0.0         | Speed limit (70km/h)  |

For the first image, the model is very sure that this is a speed limit sign of 70km/h, and the network checks it correctly. The top five soft max probabilities were:

| Probability |     Prediction	      | 
|:-----------:|:---------------------:| 
| 1.0         | Speed limit (70km/h)  |
| 0.0         | Speed limit (20km/h)  |
| 0.0         | Speed limit (30km/h)  |
| 0.0         | Speed limit (50km/h)  |
| 0.0         | Speed limit (60km/h)  |

From the above result, the network seems to be incapable of discriminating speed limit sign of 50km/h and 60km/h. This is because this stop signs have little bit slighter fonts and thicker rim. Especially, slighter font will get more hard to discriminate when the image is downsized. In this situation, the number 5 and 6 will be more ambiguous, and it seems that my network cannot discriminate it.