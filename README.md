## Self-Driving Car Engineer Nanodegree

## 2: Traffic Sign Classifier

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

Unfortunately, the data is not uniformly distributed. There are a few traffic signs that have less than 250 samples in the training set, while some traffic signs have more than 1500 samples. This usually means that the network we trained will be good at identifying those signs that have a lot of samples in the training set, and not good at identifying those signs that have less samples.

The distribution of all 3 sets looks similar. We expect the validation and test results to also be similar.
