# Pose Estimation 

<p align="center">
<img src="https://github.com/akmeraki/Semantic_Segmentation/blob/master/images/Gif.gif">
</p>


### Overview
The objective of this project is to make a Convolutional Neural Network do a pixel wise classification of multiple objects in the scene (i.e cars, pedestrians, trees, roads etc.).


### Dependencies

This project requires **Python 3.6** and the following Python libraries installed:
Please utilize the environment file to install related packages.

- [Environment File](https://github.com/akmeraki/Behavioral-Cloning-Udacity/tree/master/Environment)
- [Keras](https://keras.io/)
- [NumPy](http://www.numpy.org/)
- [Scikit-learn](http://scikit-learn.org/)
- [Tensorflow](https://www.tensorflow.org/)
- [OpenCV2](http://opencv.org/)
- [Scipy](https://www.scipy.org)

### Files in this repo
- `Semantic_Segmentation.ipynb` - The jupyter notebook used to build and train the model.
- `Notes folder` - Contains the Notes taken, Paper summaries and Documents taken.
- `Image folder` - Contains the samples images of the training data and model .
- `cityscapes.ipynb` - Jupyter notebook with some visualization and preprocessing of the Cityscape dataset.
- `helper_cityscapes.py` - python program for images pre- and  post- processing for the Cityscape dataset.

### Architecture
<p align="center">
<img src="https://github.com/akmeraki/Semantic_Segmentation/blob/master/images/fcn_arch_vgg16.png">
</p>

A Fully Convolutional Network (FCN-8 Architecture developed at Berkeley, see [paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) ) was applied for the project. It uses VGG16 pretrained on ImageNet as an encoder.
Decoder is used to upsample features, extracted by the VGG16 model, to the original image size. The decoder is based on transposed convolution layers.

The goal is to assign each pixel of the input image to the appropriate class (road, backgroung, etc). So, it is a classification problem, that is why, cross entropy loss was applied.


### How to Run the Model
You can clone the repository. Then install the required dependencies. Open a jupyter lab or jupyter notebook console and Implement the notebook- `Semantic_Segmentation.ipynb`.


### About the Dataset
This Model was trained using a dataset size of 5000 images.
Link to the cityscapes Dataset: https://www.cityscapes-dataset.com/examples/#fine-annotations
