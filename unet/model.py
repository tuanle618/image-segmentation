import torch
import torch.nn as nn
import torch.nn.functional as F
from unet.layers import DoubleConv, DownConvolution, UpConvolution, OutConv

class UNet(nn.Module):
    """
    Wrapper for U-Net Model including the Contracting Path and Expansion Path
    """
    def __init__(self, n_channels=3, n_class=2, bn=True, padding=1, upsampling=True):
        """
        Instantiates the U-Net model according to the paper
        "U-Net: Convolutional Networks for Biomedical Image Segmentation" (https://arxiv.org/pdf/1505.04597.pdf)
        :param n_channels: Channels of input image. Defaults to 3 for RGB images
        :param n_class: Class for segmentation. Defaults to 2 to train a segmentation model to extract foreground from
                        background.
        :param bn: Boolean whether or not batch-normalization should be used in the DoubleConvolution operation
        :param padding: Integer 0/1 to apply same padding (1) or valid padding (0). If valid padding (0) is used, the
                        output segmentation image has smaller size than original input image. Defaults to 1.
        :param upsampling: Boolean to use the classical bilinear upsampling followed by a convolution operation or to
                            use the transposed convolution operation. Note that the first one is the default and contains
                            less learnable parameters for those layers.
        """

        super().__init__()
        self.n_channels = n_channels
        self.n_class = n_class
        self.upsampling = upsampling

        # Start with double convolution layer without reducing height and width but only feature-map dim
        # here: input image [3, 256, 256]
        self.contract = DoubleConv(n_channels, 64, padding, bn)  # Out [64, 256, 256]

        # Build contraction layers
        # here: input is output of doubleconv layer. With downconv1 the contraction with maxpooling starts.
        self.downconv1 = DownConvolution(64, 128, padding, bn)  # Out [64, 128, 128]
        self.downconv2 = DownConvolution(128, 256, padding, bn)  # Out [128, 64, 64]
        self.downconv3 = DownConvolution(256, 512, padding, bn)  # Out [256, 32, 32]
        self.downconv4 = DownConvolution(512, 1024, padding, bn)  # Out [512, 16, 16]

        # Build expansion layers ; here: input concatenated feature map [512+512, 32, 32]
        self.upconv1 = UpConvolution(1024, 512, padding, bn, upsampling)  # Out [256+256, 32, 32]
        self.upconv2 = UpConvolution(512, 256, padding, bn, upsampling)  # Out [128+128, 64, 64]
        self.upconv3 = UpConvolution(256, 128, padding, bn, upsampling)  # Out [64+64, 128, 128]
        self.upconv4 = UpConvolution(128, 64, padding, bn, upsampling)  # Out [32+32, 256, 256]

        # Add output segmentation layer
        self.out_segementation = OutConv(64, n_class) # Out [2, 128, 128]

    def forward(self, x):
        # Compute contractions parts
        x1 = self.contract(x)
        x2 = self.downconv1(x1)
        x3 = self.downconv2(x2)
        x4 = self.downconv3(x3)
        x5 = self.downconv4(x4)

        # Compute expansion parts
        x = self.upconv1(x5, x4)
        x = self.upconv2(x, x3)
        x = self.upconv3(x, x2)
        x = self.upconv4(x, x1)
        x = self.out_segementation(x)
        return x

"""
device = torch.device('cuda:0')
u_net = UNet(n_channels=3, n_class=2, bn=True, padding=1, upsampling=True).to(device)
x = torch.randn(4, 3, 256, 256, device=device)
y = u_net(x)
"""