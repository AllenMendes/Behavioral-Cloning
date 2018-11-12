# **Behavioral Cloning Project Writeup** 

**Objectives:**
The goals / steps of this project are the following:
1. Use the simulator to collect data of good driving behavior
2. Build, a convolution neural network in Keras that predicts steering angles from images
3. Train and validate the model with a training and validation set
4. Test that the model successfully drives around track one without leaving the road
5. Summarize the results with a written report
---
### Model Architecture and Training Strategy

#### 1. Solution Design Approach and Model Architecture

I used the famous NVIDIA's CNN model architecture for this project as it has been developed specifically for such a task i.e. a machine learning model which can learn how to drive a car based on the image and steering angle data. The model consists of a convolution neural network with three 5x5 filters followed by two 3x3 filters and 3 fully connected layers at the end to produce a vehicle control parameter like steering angle from input images of size 160x320x3 (cropped to 65x320x3) 

The model include ELU layers at each convolutional layer to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer. The top 70 pixels and bottom 25 pixels are also cropped out using a Keras Cropping layer to remove the unnecessary background elements like sky, trees, grass, etc. that may distract the model from understanding the curvature of the road lane lines. This step reduces the size of the input images to 64x320x3.  

The model contains dropout layers with a drop out rate of 50% in between the convolutional layers and also between the fully connected layers in order to reduce overfitting. I trained the model in small bacth sizes of 32 with a MSE (Mean Sqaure Errors) as the loss function and an Adam optimizer. At the end, the total traianble paramters were 348,219 for this specific setup.

Here is a visualization of the architecture:
![model](https://github.com/AllenMendes/Behavioral-Cloning/blob/master/nvidia_network.png)

#### 2. Creation of the Training Set & Training Process

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
