## Self-Driving Car Engineer Nanodegree

## Project 2: Traffic Sign Classifier

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

#### Model Architecture

The neural network architecture used in the project is based on the [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.

The LeNet architecture is a well known architecture for image recognition. "LeNet is small and easy to understand — yet large enough to provide interesting results." That's the reason we choose LeNet as the first architecture to try on the Traffic Signs data set.

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

#### Solution Approach

The final model results were:

1. validation set accuracy = 95.9%
2. test set accuracy = 94.9%

### Test a Model on New Images

#### Acquiring New Images

Five new German Traffic signs were downloaded from the web, cropped and resized to 32x32.
