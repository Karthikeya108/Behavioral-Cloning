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

The overall strategy for deriving a model architecture was to begin with a simple model and add the layers to the network to make it deep.

1. A simple neural network with a single hidden layer trained on a single simulated drive data. When tested, the car would drive out of the track within few seconds.
2. Simulated more data. Drove in reverse direction and covered edge case for instance driving into the center of the road when the car at the edge of the road. Collected drive data from steep curves.
3. I split the data into the training and validation set in order to observe the performance and detect overfitting.
4. Built a bigger network very similar to LeNet and trained the model on the data. 
5. On testing, car would drive for a while but would drive away from the track or crash into something at the steep curves of the road.
6. Added few more convolutional layers to the network.
7. Trained the model for more epochs
8. Added dropout layers to avoid overfitting and also loaded more edge case data of steering the car on to the center of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road on track 1. 

Next, I collected the data for track 2 and used the same model. It gave good results with approriate amount of training data. However, one problem I could not solve on track 2 was that the car would come to a halt (Speed 0) and would not start driving again.

#### 2. Final Model Architecture

The final model architecture (model.py lines 31-46) consisted of a convolution neural network with the following layers:

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

#### 3. Creation of the Training Set & Training Process

1. Normal driving - Driving on the center of the road - multiple laps.
2. Driving in reverse direction - Driving on the center of the road.
3. Covering the edge case scenarios - Steering back the car from edge of the track. Quick steering away from objects near steep curvings of the road etc.

I normalized the image data before feeding it to the network.

I finally randomly shuffled the data set and put 10% of the data into a validation and test sets. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20. I used an adam optimizer so that manually training the learning rate wasn't necessary.
