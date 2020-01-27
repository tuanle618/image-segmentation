## The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation (2016) ##
### Simon Jégou, Michal Drozdzal, David Vazquez, Adriana Romero, Yoshua Bengio ###
#### Link: https://arxiv.org/abs/1611.09326 ####

This repository contains the Fully-Convolutional DenseNet (FC DenseNet) pytorch implementation.  
The FC DenseNet model architecture is depicted below:  
![alt text](https://github.com/tuanle618/image-segmentation/blob/master/tiramisu/architecture.png "FC DenseNet Architecture")    
One `DenseBlock` is defined as stacking several (`growth_rate`) layers and concatenate each output as depicted in the upcoming figure:  
![alt text](https://github.com/tuanle618/image-segmentation/blob/master/tiramisu/denseblock.png "DenseBlock illustration").  
Here, each `Layer` consists a composition of [BatchNorm - ReLU - Conv - Dropout].  
The basic building modules are implemented in `layers.py` and depicted in the figure below.  
![alt text](https://github.com/tuanle618/image-segmentation/blob/master/tiramisu/layers.png "Layers in FC DenseNet").

  
 
#### Options ####
- Custom modules can be edited in `layers.py`.
- The model can be modified in `model.py`.
