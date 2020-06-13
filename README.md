# 2D-to-3D Video Conversion with CNNs

## Setup
Install MXNet following the official document.
Open mxnet/config.mk and ```set USE_CUDA to 1``` and ```USE_CUDNN to 1```. 
Append ```EXTRA\_OPERATORS=path/to/Plus-1D/operators``` to path/to/mxnet/config.mk and recompile MXNet

## Dependencies
*   CUDA 7.0+
*   CUDNN 4+

## Workflow
![alt text](https://raw.githubusercontent.com/piiswrong/deep3d/master/img/teaser.png)


## Model performance outputs

![alt text](https://raw.githubusercontent.com/piiswrong/deep3d/master/img/1_GIF.gif) ![alt text](https://raw.githubusercontent.com/piiswrong/deep3d/master/img/8_GIF.gif)
![alt text](https://raw.githubusercontent.com/piiswrong/deep3d/master/img/3_GIF.gif) ![alt text](https://raw.githubusercontent.com/piiswrong/deep3d/master/img/4_GIF.gif)
![alt text](https://raw.githubusercontent.com/piiswrong/deep3d/master/img/5_GIF.gif) ![alt text](https://raw.githubusercontent.com/piiswrong/deep3d/master/img/6_GIF.gif)
![alt text](https://raw.githubusercontent.com/piiswrong/deep3d/master/img/7_GIF.gif) ![alt text](https://raw.githubusercontent.com/piiswrong/deep3d/master/img/2_GIF.gif)

## Background
3D imagery has two views, one for the left eye and the other for the right.
To convert an 2D image to 3D, you need to first estimate the distance from camera for each pixel (a.k.a depth map) and then wrap the image based on its depth map to create two views.

The difficult step is estimating the depth map. For automatic conversion, we would like to learn a model for it.
There are several works on depth estimation from single 2D image with DNNs. However, they need to be trained on image-depth pairs which are hard to collect. As a result they can only use small datasets with a few hundred examples like NYU Depth and KITTI. Moreover, these datasets have only static scenes.

In Contrast, Plus-1D can be trained directly on 3D movies that have tens of millions frames in total.
We do this by making the depth map an internal representation instead of the end prediction.

Thus, instead of predicting an depth map and then use it to recreate the missing view with a separate algorithm, we train depth estimation and recreate end-to-end in the same neural network.

Here are some visualizations of our internal depth representation to visualise it's working:

![alt text](https://raw.githubusercontent.com/saswat0/Plus-1D/master/img/0059.jpg)
![alt text](https://raw.githubusercontent.com/saswat0/Plus-1D/master/img/0112.jpg)
![alt text](https://raw.githubusercontent.com/saswat0/Plus-1D/master/img/0131.jpg)
![alt text](https://raw.githubusercontent.com/saswat0/Plus-1D/master/img/0163.jpg)
![alt text](https://raw.githubusercontent.com/saswat0/Plus-1D/master/img/0203.jpg)
![alt text](https://raw.githubusercontent.com/saswat0/Plus-1D/master/img/0266.jpg)
![alt text](https://raw.githubusercontent.com/saswat0/Plus-1D/master/img/0351.jpg)
![alt text](https://raw.githubusercontent.com/saswat0/Plus-1D/master/img/0459.jpg)
![alt text](https://raw.githubusercontent.com/saswat0/Plus-1D/master/img/0471.jpg)

Following each image, there are 4x3 maps of depth layers, ordered from near to far. You can see that objects that are near to you appear in the first depth maps and objects that are far away appear in the last ones. This shows that the internal depth representation is learning to infer depth from 2D images without been directly trained on it.

## Code
This work is done with [MXNet](https://github.com/dmlc/mxnet), a flexible and efficient deep learning package. The trained model and a prediction script is in Plus_1D.ipynb.

## References
[Deep3D](https://github.com/piiswrong/deep3d)