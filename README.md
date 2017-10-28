## Self-Driving Car Engineer Nanodegree

## Project 2: Traffic Sign Classifier

Project Jupyter Notebook: Traffic_Sign_Classifier.ipynb

Notebook HTML snapshot: report.html

Test Images:
1. traffic-1.jpg
2. traffic-2.jpg
3. traffic-3.jpg
4. traffic-4.jpg
5. traffic-5.jpg

### Introduction

The goals / steps of this project are the following:

1. Load the data set
2. Explore, summarize and visualize the data set
3. Design, train and test a model architecture
4. Use the model to make predictions on new images
5. Analyze the softmax probabilities of the new images
6. Summarize the results with a written report

### Data Set Summary & Exploration

Using python and numpy methods to calculate summary statistics of the traffic signs data set:


1. The size of training set is 34799
2. The size of the validation set is 4410
3. The size of test set is 1263
4. The shape of a traffic sign image is (32, 32, 3)
5. The number of unique classes/labels in the data set is 43

### Exploratory visualization of the dataset

This graph shows the traffic sign distribution of the training, validation and test set.

![Label Distribution](https://github.com/ongchinkiat/SDCND-Project2/raw/master/labels-bar-graph.jpg "Label Distribution")

Unfortunately, the data is not uniformly distributed. There are a few traffic signs that have less than 250 samples in the training set, while some traffic signs have more than 1500 samples. This usually means that the network we trained will be good at identifying those signs that have a lot of samples in the training set, and not so good at identifying those signs that have less samples.

The distribution of all 3 sets looks similar. We expect the validation and test results to also be similar.

### Design and Test a Model Architecture

#### Pre-process the Data Set

For the image data, a simple normalize() function is used to normalize the data to between -1 and 1.

```
def normalize(x):
    y = ((x - 128.0)/ 128.0) - 1
    return y
```

Since the result of the network training turns out to be ok, no further pre-processing steps were done on the data set.

#### Model Architecture

The neural network architecture used in the project is based on the [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.

The LeNet architecture is a well known architecture for image recognition. "LeNet is small and easy to understand — yet large enough to provide interesting results." That's the reason we choose LeNet as the first architecture to try on the Traffic Signs data set.

In addition, 3 drop out layers were used between the fully connected areas to improve performance of the network.

The details of the model is as follows:

| Layer | Description |
| ----- | ----- |
| Input | 32x32x3 RGB Image |
| Convolution 5x5 | 1x1 stride, Valid padding, output 28x28x30 |
| RELU |   |
| Max pooling |	2x2 stride, outputs 14x14x30
| Convolution 5x5 | 1x1 stride, Valid padding, output 10x10x64 |
| RELU |   |
| Max pooling |	2x2 stride, outputs 5x5x64 |
| Flatten | outputs 1600 |
| Fully Connected | outputs 1024 |
| RELU |   |
| Drop Out | Keep Probability = 0.5  |
| Fully Connected | outputs 512 |
| RELU |   |
| Drop Out | Keep Probability = 0.5  |
| Fully Connected | outputs 256 |
| RELU |   |
| Drop Out | Keep Probability = 0.5  |
| Fully Connected | outputs 43 (number of classes) |

#### Model training

The model is trained using these hyperparamters:

1. epochs = 50
2. batch size = 512
3. learning rate = 0.001
4. keep probabilities = 0.5
5. optimizer = AdamOptimizer

These hyperparamter values are the initial values I chose. I found that changing the number of fully connected layers, adding the drop out layers, and changing the number of neurons in each layer has a greater impact on the performance of this network.


#### Solution Approach

The final model results were:

1. validation set accuracy = 95.5%
2. test set accuracy = 94.8%

### Test a Model on New Images

#### Acquiring New Images

Five new German Traffic signs were downloaded from the web, cropped and resized to 32x32.

The test images are:
1. traffic-1.jpg ![Test 1](https://github.com/ongchinkiat/SDCND-Project2/raw/master/traffic-1.jpg "Test 1")

2. traffic-2.jpg ![Test 2](https://github.com/ongchinkiat/SDCND-Project2/raw/master/traffic-2.jpg "Test 2")

3. traffic-3.jpg ![Test 3](https://github.com/ongchinkiat/SDCND-Project2/raw/master/traffic-3.jpg "Test 3")


4. traffic-4.jpg ![Test 4](https://github.com/ongchinkiat/SDCND-Project2/raw/master/traffic-4.jpg "Test 4")


5. traffic-5.jpg ![Test 5](https://github.com/ongchinkiat/SDCND-Project2/raw/master/traffic-5.jpg "Test 5")

The 5 test images are taken from various sources, and are originally parts of much larger images. They are cropped and resized to 32x32, with the traffic sign taking up more the half the area of the whole image. This make them structurally similar to our training images. So we'll expect the model to at least identify some of the images correctly.

#### Performance on New Images

| Filename | Image | Correct Label | Correct Description | Prediction Label | Prediction Description |
| ----- | ----- | ----- | ----- | ----- | ----- |
| traffic-1.jpg | ![Test 1](https://github.com/ongchinkiat/SDCND-Project2/raw/master/traffic-1.jpg "Test 1") | 1 | Speed limit (30km/h)| 1 | Speed limit (30km/h) |
| traffic-2.jpg | ![Test 2](https://github.com/ongchinkiat/SDCND-Project2/raw/master/traffic-2.jpg "Test 2") | 33 | Turn right ahead | 33 | Turn right ahead |
| traffic-3.jpg | ![Test 3](https://github.com/ongchinkiat/SDCND-Project2/raw/master/traffic-3.jpg "Test 3") | 17 | No entry | 17 | No entry |
| traffic-4.jpg | ![Test 4](https://github.com/ongchinkiat/SDCND-Project2/raw/master/traffic-4.jpg "Test 4") | 25 | Road work | 36 | Go straight or right |
| traffic-5.jpg | ![Test 5](https://github.com/ongchinkiat/SDCND-Project2/raw/master/traffic-5.jpg "Test 5") | 14 | Stop | 14 | Stop |

Predictions of images 1,2,3,5 is correct. Accuracy is 80%

A test of only 5 images is too few to make any meaningful conclusion on the results. This performance on the new images cannot be meaningfully compared to the accuracy results of the test set.

#### Model Certainty - Softmax Probabilities

The top 5 softmax probabilities for each image is also printed in the Jupyter Notebook.

For images 1-4, the top prediction result has a softmax probability of 1.0000, with the rest of the probabilities all lower than 0.0001.

For image 5, the top prediction result has a softmax probability of 0.937, the 2nd 0.039, 3rd 0.022, and the rest lower than 0.001.
