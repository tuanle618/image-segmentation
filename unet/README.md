## U-Net: Convolutional Networks for Biomedical Image Segmentation (2015) ##
### Olaf Ronneberger, Philipp Fischer, Thomas Brox ###
#### Link: https://arxiv.org/abs/1505.04597 ####

This repository contains the U-Net pytorch implementation.  
The U-Net model architecture is decicted below:
![alt text](https://github.com/tuanle618/image-segmentation/blob/master/unet/architecture.png "U-Net Architecture")
  
 
#### Options ####
- The `DoubleConv` layer contains batchnormalization as opposed in the original paper.
- The default U-Net model supports padding such that the output of each DoubleConv has the same spatial resolution as its input.
- Custom modules can be edited in `layers.py`.
- The model can be modified in `model.py`.
