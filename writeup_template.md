# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: examples/all_classes.png “All Classes”
[image2]: examples/augmented_images.png “Augmented data”
[image3]: examples/Test_images.png “Test set images”
[image9]: examples/traffic_sings_distribution.png “Training distribution”
[image10]: examples/preprocess_images.png “augmented data”

## Rubric Points

---
### Submission files

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

The data provided in the project comprises of German traffic signs.

* Number of training examples = 34799
* Number of testing examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43


#### Distribution of training data

Here is an exploratory visualization of the Training data. 

![alt text][image9]
![alt text][image1]

### Design and Test a Model Architecture

#### Data preparation

* First step: I convert images into grayscale to remove any discrepancy due to color in original images. 
* Feature normalization: All the pixel intensities are normalized to have better convergence of gradient decent algorithm.  
* Data augmentation : Added additional data as the distribution of images in training set is not completely uniform. 
    ** Image translation 
    ** image rotation 
    ** Gaussian filter 
    ** Bluring image

The sample images after data augmentation as follows:

![alt text][image10]

#### Model architecture and hyper-parameter info
I used a standard Lenet architecture. 
My final model consisted of the following layers:

| Layer         			|     Description	        						| 
|:---------------------:	|:---------------------------------------------:	| 
| Input         			| 32x32x3 RGB image   							| 
| Convolution 5x5x6     	| 1x1 stride, same padding, outputs 28x28x6 		|
| RELU					|											|
| Max pooling	      		| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5x6x16	    	| outputs 10x10x16      						|
| RELU					|											|
| Max pooling	      		| 2x2 stride,  outputs 5x5x16 					|
| Flatten					| output - 400        							|
| Dropout					|        										|
| Fully connected			| output - 120        							|
| Dropout					|         									|
| Fully connected			| output - 84      							|
| Dropout					|        										|				
| Fully connected			| output - 43      							|
| softmax					|											|
|						|											|
 


##### Hyper-Parameter
* Epochs = 30
* Batch_size = 128
* Rate = 0.001
* mean of initial random weights = 0
* Variance of initial random weights = 0.1
* Optimizer : Adam optimizer




## Structuring Machine learning problem

1. Start with known LeNet architecture based CNN.
2. In this classification problem, it is fair to assume human error is very low. However, I looked at the data from training set and concluded that some of the images are hard to recognize by me. Hence, it is reasonable to have some training error.
3. Performed error analysis on training and validation data. 
4. Initially the training error was low and validation error was high. This made me focus on trying to solve the problem of overfitting.
5. Individually tried the following ideas
	1. Increase data using data augmentation methods.
	2. Use dropout technique to reduce variance
	3. Increase number of epochs.

 4. Problem approach details:

My final model results were:
* training set accuracy of 0.991
* validation set accuracy of 0.944
* test set accuracy of 0.915

If an iterative approach was chosen:
* Initially used a LeNet CNN on training data. With 10 Epochs, and learning rate = 0.001. Converted raw data into gray scale image and normalized training data.
   * Training error = 0.992
   * Validation error = 0.902
* I observed the images in the training set and found few of the images hard to recognize. Hence was satisfied by the Training error.
* Focused on the problem of overfitting. 
   * Firstly, used Dropout with 0.5 probability for each node .(By keeping rest of the parameters same)
   * Training Accuracy = 0.980
   * Validation Accuracy = 0.947
   * Test accuracy  = 0.925

* Next I ran the model on 5 images from internet and saw an accuracy of 40%. Observed that an image corresponding to no vehicles was not being detected correctly. 

![alt text][image3]

*softmax output of five test images with labels as (14,15,12,36,4)
INFO:tensorflow:Restoring parameters from ./lenet_more_data
TopKV2(values=array([[  7.98735321e-01,   1.00402549e-01,   6.97058365e-02,
          8.34206026e-03,   5.77890780e-03],
       [  8.30667198e-01,   7.19514564e-02,   6.70547485e-02,
          2.44841129e-02,   5.68909571e-03],
       [  9.94002640e-01,   1.26411638e-03,   1.18246512e-03,
          7.61133444e-04,   5.43891161e-04],
       [  7.28698611e-01,   2.10569084e-01,   4.26086113e-02,
          1.05189132e-02,   7.00206682e-03],
       [  6.32534981e-01,   2.14114100e-01,   7.33823031e-02,
          1.84143130e-02,   9.91379377e-03]], dtype=float32), indices=array([[14, 13, 15, 33, 39],
       [35,  3, 13, 34, 15],
       [12, 33, 40,  9, 38],
       [41, 32, 36, 20, 28],
       [25,  8, 39,  4,  7]], dtype=int32))

Model prediction results (14,35,12,41,25)

Noticed that the corresponding number of images in training set are lower in number. Hence, decided to add more data so that final distribution of data in training set has certain minimum number of samples for each classes. 

* reran the model with more data. However, I was also thinking that this could further reduce our variance problem as the amount of training data is increasing.

* increased the number of epochs to further reduce variance.


### Analysis of Model on Test images

My model has a tough time recognizing harder images like pedestrians, road work. Any suggestions u are most welcome. 