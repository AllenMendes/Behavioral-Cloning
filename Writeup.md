# **Behavioral Cloning Project Writeup** 

The setup includes a simulator environment provided by Udacity with a car that has 3 cameras mounted on the left, center and right locations of the front facing windshield/hood, which also records the steering angle while driving around in the simulated environment. First, the user drives the car around on a track with straight and curve lanes with different textures of the road and lane lines while recording the camera and steering angle data. The model then learns how to drive the car around on the same track based on the data provided resulting in a self-driving car completely driven by a AI model.

**Objectives:**
The goals / steps of this project are the following:
1. Use the simulator to collect data of good driving behavior
2. Build, a convolution neural network in Keras that predicts steering angles from images
3. Train and validate the model with a training and validation set
4. Test that the model successfully drives around track one without leaving the road
5. Summarize the results with a written report
---
## Model Architecture and Training Strategy

#### 1. Solution Design Approach and Model Architecture

I used the famous NVIDIA's CNN model architecture for this project as it has been developed specifically for such a task i.e. a machine learning model which can learn how to drive a car based on the image and steering angle data. The model consists of a convolution neural network with three 5x5 filters followed by two 3x3 filters and 3 fully connected layers at the end to produce a vehicle control parameter like steering angle from input images of size 160x320x3 (cropped to 65x320x3) 

The model include ELU layers at each convolutional layer to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer. The top 70 pixels and bottom 25 pixels are cropped out using a Keras Cropping layer to remove unnecessary background elements like sky, trees, grass, etc. that may distract the model from understanding the curvature of the road lane lines. This step reduces the size of the input images to 64x320x3.  

The model contains dropout layers (with a drop out rate of 50%) in between the convolutional layers and also between the fully connected layers in order to reduce overfitting. I trained the model in small batch sizes of 32 with a MSE (Mean Sqaure Errors) as the loss function and an Adam optimizer. At the end, the total traianble paramters were 348,219 for this specific setup.

Here is a visualization of the architecture:

![model](https://github.com/AllenMendes/Behavioral-Cloning/blob/master/nvidia_network.png)

I also implemented the data generator function using the "Yield" command. It made sense creating this function as I didn't want to store such large number of images in memory while the model is being trained. The generator function instead gives out only a portion of the dataset (batch size 32) while the model is being trained during each stage of a single epoch. 

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving, with occasional maneuvers of the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the center of the lane if it starts drifting off to either of the edges of the road.

To augment the recorded dataset, I performed the following image processing techniques:

1. **Random Brightness**
To simulate variation of lighting conditions on the road, I performed random changes of brightness of the image by first converting the RGB image to its HSV equivalent and then randomly changing the Hue and Saturation values to brighten or darken the overall image intensity.

2. **Random Shadow**
To simulate random patches of shadow on the road due to a tree or cloud cover or by adjacent vehicles/objects on the road, I first converted the image into the HLS equivalent and then selected a random shape in the image dimensions of 160x320 pixels to be a shadowed portion and changed its Hue and Light value to a random value.

3. **Image and steering angle flipping**
I also flipped the center image and the steering angle value to augment more data and to further generalize the model. I also added a 25% correction to the images obtained from the left and right cameras so that the model aggressively takes harder turns at the edges if the car is close to either of the edges of the road.

At the end I had 4872 images to train on, out of which I separated this dataset into two parts: 3897 training images and 975 validation images.

I created an AWS EC2 instance on the cloud and uploaded my dataset along with the main code - [Behavioral_Cloning.py](https://github.com/AllenMendes/Behavioral-Cloning/blob/master/Behavioral_Cloning.py). I trained the model over 5 epochs and observed that the model loss went from 0.0783 to 0.0384 (gradual decrease with no sudden spikes) which is good indicator that the model is not over fitting the data.

#### 3. Final result and conclusion
After few hours of the training the model, the final trained model was generated - [modelRecovery.h5](https://github.com/AllenMendes/Behavioral-Cloning/blob/master/modelRecovery.h5)

I loaded the model into the simulator using the drive.py file and the car started driving by itself using real time data from the cameras and completely controlled by the trained model !

### Here is the link to the final output video of track 1 : [Video](https://drive.google.com/file/d/1aVgKcRkcEN7ZQdXAE8SMCLM68HEGTC-A/view?usp=sharing)

Conclusions:
1. Due to lack of time, I could not recorded enough data to generalize the model even further. But I am sure that collecting more data and augmenting it even further will make by model much much better !
2. Data augmentation techniques really help train the model for several tricky situations like shadows, uneven road elevations, etc.
3. The dataset should contain a fairly even spread of data in terms of steering angles so that the model doesn't lean towards the data which is mostly available in the dataset. For example, if the dataset has majorly left angle steering data and very few right angle steering data, the model may not be able to steer the car to the right if it sees a huge right curve/turn just because it doesn't know how to handle such a turn.

