# Mask_Detection
Face Mask Detection using Python, Keras, OpenCV

## Import Statements -
1. Sequential is a prebuild Keras model where you can add layers.
2. Conv2D layer is used to extract the features from the image. Convolution helps with the sharpening face detection and other operation that can help the machine to learn specific characteristics of an image.
3. Max Pooling 2D layers helps to reduce the size of the data or we can say it reduces image dimensionality without importing features or patterns.
4. Dropout layer is to to reduce overfitting.
5. Flatten layer it transforms two dimensional matrix of features into a vector that can be fed into fully connected to neural network classifier. So that means converting data into one dimensional array for inputting into next layer. So we flatten the output of convolution layer to create a single long feature.
6. Dense layer is a classic fully connected neural network layer where each input node is connected to each output node.
7. To compile the model we will need an optimizer, so we had used 'adam' optimizer.
8. The image data generator class is used for augmentation purpose. So augmentation is one useful technique in building CNN that can increase the size of training set without acquiring new images. So the idea over here is very simple, just duplicate images with some kind of variation so model can learn from more examples.

__Link to Dataset__

https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset

## Overview Structure of Directories - 
The New Masks Dataset contains 4 directories out of which 3 are from the original dataset on Kaggle, the 4th directory *Static Images* was created later  which contains 2 images randomly downloaded from the Internet to check if our model can accurately make the right prediction or not.

The three folders inlude -
- Train
- Validation
- Test

Each of these three folders have sub-folders - Mask and Non Mask.

## Image visualization using matplotlib - 
So we are going to create a grid of 16 images. So we will visualize 8 images from te mask directory & will visualizing another 8 images fron the non_mask directory of training dataset.

So overall, we will get a grid of 16 images where the number of columns and the number of rows will be 4.

## Image Augmentation - 
Data augmentation is the process when you create a new data based on the modifications of our existing data.
So basically, we are creating a new documented data by making resonable modification on our data in the training data.
It helps you to avoid overfitting, so it is often used when you have limited data.

Code Example - 
train_augmentation = ImageDataGenerator(rescale = 1/255,zoom_range = 0.2,rotation_range = 40, horizontal_flip = True, vertical_flip = True)
rescale = used to normalize 
zoom_range = 0.2 -- zooming into random parts of the images by 20%
rotation_range = 40 degrees -- rotate the image of 40 degrees.

To generate actual generator function, we will call method from the variables we created as they are type of Image Data Generator.

Code - 
train_generator = train_augment.flow_from_directory(train_directory,target_size=(200,200),batch_size=32,class_mode='binary')
.flow_from_directory - to load the images
Parameters - 
1) train_directory - Directory where mask and no mask folders are present
2) target_size = Dimension we want to rescale our image to.
3) batch_size = How many images we will feed to a network at a time.
4) class_mode = 'binary' ---> Class_mode is binary in our case since we just have to o/p whether the person is wearing a mask or not.

# Building CNN Model - 
Conv2D preserves a relationship between pixels by learning image features, so the parameter says to extract 32 features from input image.
We have kept the parameter padding as same to not to lose information of an image.
Adding Max pooling layer, to reduce the image.
Adding Dropout layer to drop 50% of neurons to avoid overfitting.


