**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py : containing the script to create and train the model on Track-1 of the Simalator
* model-track-2.py : containing the script to create and train the model on Track-2 of the Simalator
* drive.py : for driving the car in autonomous mode
* model-PSAModel-track-1a.h5 : containing a trained convolution neural network on Track-1 of the Simalator
* model-PSAModel-track-2.h5  : containing a trained convolution neural network on Track-2 of the Simalator
* README.md : summarizing the results
* video-track-1.mp4 : sample autonomous test drive using the trained model on Track-1
* video-track-2.mp4 : sample autonomous test drive using the trained model on Track-2

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python3 drive.py model-PSAModel-track-1a.h5
```
NOTE: Used Python version : 3.5.2 and the Keras version : 2.1.6. The code did not work with Python version 3.6

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model. Although the code is self explanatory, I have added few comments.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The data is used with all the 3 channels as the input to the network. The data is normalized as shown on the line 32.

Network architecture us as follows: (Started with the LeNet model and modified to reach the following)


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 array   							| 
| Convolution 5x5x6     	| 1x1 stride, valid padding 	|
| RELU					|												|
| Max pooling	      	| 2x2 pool size				|
| Convolution 5x5x16	    | 1x1 stride,  valid padding      									|
| RELU					|												|
| Max pooling	      	| 2x2 pool size				|
| Convolution 5x5x16	    | 1x1 stride,  valid padding      									|
| RELU					|												|
| Max pooling	      	| 2x2 pool size				|
| Convolution 5x5x16	    | 1x1 stride,  valid padding      									|
| RELU					|												|
| Max pooling	      	| 2x2 pool size				|
| Flatten         |  
| Dropout         | drop rate 0.4       |
| Fully connected (Dense)		| outputs 120        									|
| Dropout         | drop rate 0.2       |
| Fully connected (Dense)		| outputs 84        									|
| Fully connected (Dense)		| outputs 1        									|

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 42 & 44). Various driving simulations were used to generate more data. The edge case scenario data was also generated and loaded multiple times to have a significant effect on the model learning. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 19 / 105).

#### 4. Appropriate training data

Multiple simulation was run to generate the data. 

1. Normal driving - Driving on the center of the road.
2. Driving in reverse direction - Driving on the center of the road.
3. Covering the edge case scenarios - Steering back the car from edge of the track. Quick steering away from objects near steep curvings of the road etc.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
