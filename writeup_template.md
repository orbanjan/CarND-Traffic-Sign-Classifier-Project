# **Traffic Sign Recognition** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/1.jpg "Traffic Sign 1"
[image5]: ./examples/2.jpg "Traffic Sign 2"
[image6]: ./examples/3.jpg "Traffic Sign 3"
[image7]: ./examples/4.jpg "Traffic Sign 4"
[image8]: ./examples/5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/orbanjan/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribution looks like for the train, test, validation set. It seems that they are from the same distrubution. Some trafic sign occurance are very high compering to others (2000 vs 100) which indicates that some augmentation might help especially on low-occurancy items.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Originally  I decided to use grayscale but it did not give high accuarcy at first. Further investigation needed why.
Greyscale images would be further enhanced for saturation, contrasts and edges as well.
Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it will give data around zero with +/ 0.5.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| C1: Convolution (5x5)     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| S2: Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| C3: Convolution (5x5)     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| S4: Max pooling	      	| 2x2 stride,  outputs 5x5x16  				|
|	Flatten					|	400								|
| C5: Fully connected		|  120       									|
| RELU				|        									|
|	Dropout					|		0.7							|
| F6: Fully connected		|  84     									|
| RELU				|        									|
|	Dropout					|		0.7							|
| OUTPUT		|  43     									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer. Batch size was = 256 and the number of epochs = 30. Learning rate was = 0.001
In AWS Instance I used epochs =100 and with different learning rates from 0.1 to 0.0001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.939 
* test set accuracy of 0.931

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen? LeNet 5 with RGB. Originally I choosed grayscale image but accuracy was very low.
* How was the architecture adjusted and why was it adjusted? I choosed 2 dropout at 0.5 rate at the 2 fully connected layer because there was a big differernce between validation and testing due to overfitting.
Which parameters were tuned? How were they adjusted and why? keep_probability was between 0.5-0.8 -> 0.7 with reasonable good results.
* Why did you believe it would be relevant to the traffic sign application? It could be trained well within reasonale time with huge dataset.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? Training accuracy was close to 1.0 and the testing and validation was very close > 0.93.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Pls note that the original files are also inlcuded in *.ppm format but md file did not show.(Showing in BGR) Some of the images might be difficult to classify because of low contract and darkness. Also some objects are also hiding some portion of the signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 50km/h      		| Speed limit 50km/h  									| 
| Speed limit 80km/h      		| Speed limit 80km/h  									| 
| Go straight or right				| Go straight or right											|
| Stop      		| Stop					 				|
| General Caution			| General Caution      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.93

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the last 2 cells of the Ipython notebook.

For the first image, the model is absolutely sure that this is a speed limit 50km/h (probability of ~100%), and the image does contain a speed limit 50km/h  The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0e00        			| Speed limit 50km/h  									| 
| 2.45e-10     				| Speed limit 80km/h 										|
| 8.72e-11					| Speed limit 60km/h											|
| 1.42e-15	      			| Speed imit 30 km/h		 				|
| 6.34e-25				    | Wild animals crossing      							|




### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


