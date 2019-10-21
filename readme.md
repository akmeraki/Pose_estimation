# Pose Estimation

<p align="center">
<img src="https://github.com/akmeraki/Pose_estimation/blob/master/Images/Pose_estimation.gif">
</p>


### Overview
Human pose estimation problem tackles the issue of localization of human body joints. There
are many complex architectures specifically for Human Pose Estimation problem with significant
differences but similar accuracy. Several complex architectures will still come up.

So there is a need to set a new baseline with minimal changes to existing, problem independent, architectures that are suitable for any problem in computer vision. We aim to address the problem of
estimating the poses of single and multi human instances in multiple images existing in the MS
COCO (Common Objects in Context) dataset. We explored the possibility of adding deconvolution layers at the end of a ResNet architecture. We also explored various attention based mechanisms that can provide better localization in the context of Pose Estimation.

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
- `Images` - Contains the samples images of the training data, gifs and model .
- `main` - Model file , generate batch file , Config file, train and test file.
- `lib` - Contains base resnet and other utils.
- `tool` - tool to extract pose info from COCO.
- `With Attention` - Attention mechanism added to the Network.
- `Output` - outputs the test images , model and log files.
s
### Architecture
<p align="center">
<img src="https://github.com/akmeraki/Pose_estimation/blob/master/Images/DL.png">
</p>

#### Holistic attention
<p align="center">
<img src="https://github.com/akmeraki/Pose_estimation/blob/master/Images/attention_viz_com.jpg" alt="drawing" width="600">
</p>

### Part-wise Attention
<p align="center">
<img src="https://github.com/akmeraki/Pose_estimation/blob/master/Images/pjimage.jpg"alt="drawing" width="600">
</p>

### Results

### How to Run the Model
You can clone the repository. Then install the required dependencies. Run the Train and test python files in the With Attention folder.


### About the Dataset

Link to the MS COCO Dataset:http://cocodataset.org/#keypoints-2019
