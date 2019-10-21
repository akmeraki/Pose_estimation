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

We employ ResNet-50 Architecture as our basic architecture. We add 4 deconvolution layers to the end of the
ResNet-50 architecture to produce heatmaps of 64x48 for each keypoint. To scale this architecture to multiple people we employ an object detection algorithm to predict bounding boxes for each human in an image. We then train our ResNet architecture to predict the pose from each bounding box separately. COCO dataset has bounding boxes built-in which makes it easier for us to train. While for testing, we use YOLO-V2 object detection architecture pretrained on COCO dataset images for generating bounding boxes.

#### Holistic attention
<p align="center">
<img src="https://github.com/akmeraki/Pose_estimation/blob/master/Images/attention_viz_com.jpg" alt="drawing" width="400">
</p>
Holistic attention works by attending the whole image of the human body ignoring the background. From previous research work, it is evident that more contextual information leads to better localization of key points. We achieve this by attending images in various scales. Specifically, we use attention in final 4 layers of deconvolution each of shape 8x6, 16x12, 32x24, 64x48. We use k-nearest neighbor upsampling to make each attention map of shape 64x48. We finally add these attention-maps and then attend the final deconvolution layer. Note that here although we create attention maps from each deconvolution layer, we do not attend every layer. Instead, we attend only the output from the final deconvolution layer.

By doing this we preserve contextual information from various scales as well as the information flowed through deconvolution layers. For holistic attention, we experimented with softmax and conditional random field attention. Gaussian attention is not a good candidate since we want to attend the whole human body with similar weights. Also, spatial transforms are not great candidates since it only does cropping and our network would still have non-useful background information. CRF attention in takes into consideration the local pattern spatial correlations unlike global spatial softmax. Since we want to attend continuous parts of image, softmax attention did not work as expected. CRF attention works best for holistic attention.

### Part-wise Attention
<p align="center">
<img src="https://github.com/akmeraki/Pose_estimation/blob/master/Images/pjimage.jpg"alt="drawing" width="400">
</p>
Part-wise attention works by attending only parts of the human body. This, combined with Holistic attention, works like a hierarchical attention model wherein we first attend the whole body and then attend parts of the human body. We introduce part-wise attention model by taking input from the holistic attention and then attending them for 17 key points. Gaussian attention and Spatial Transforms attention
worked as expected but the predictive power of the network did not increase significantly. We believe this is due to the fact that when localizing a keypoint, looking only at one keypoint might not help as much as looking at 2 or 3 successive or symmetric keypoints would. Although softmax attention captures this information, since it does not take spatial correlations into account, the network takes more
time than Spatial CRF to achieve comparable performance. Therefore, Spatial CRF works the best for Part-wise attention maps as well.

### Results
<p float="left">
  <img src="https://github.com/akmeraki/Pose_estimation/blob/master/Images/0.jpg" width="300" />
  <img src="https://github.com/akmeraki/Pose_estimation/blob/master/Images/221.jpg" width="300" />
  <img src="https://github.com/akmeraki/Pose_estimation/blob/master/Images/35.jpg" width="300" />
</p>


### How to Run the Model
You can clone the repository. Then install the required dependencies. Run the Train and test python files in the With Attention folder.


### About the Dataset

Link to the MS COCO Dataset:http://cocodataset.org/#keypoints-2019
