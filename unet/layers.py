import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Helper module for a double convolution layer:
    [Conv - BN - ReLU] ** 2
    Note that this implementation contains the optional zero-padding.
    """
    def __init__(self, in_channels, out_channels, padding, bn):
        super().__init__()
        if bn:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=int(padding)),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=int(padding)),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=int(padding)),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=int(padding)),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)

class DownConvolution(nn.Module):
    """
    Helper module for the contraction part of U-Net applying the DoubleConv followed by MaxPooling for downsampling.
    """
    def __init__(self, in_channels, out_channels, padding=1, bn=True):
        super().__init__()
        self.doubleconv = DoubleConv(in_channels, out_channels, padding, bn)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        return self.maxpool(self.doubleconv(x))

class UpConvolution(nn.Module):
    """
    Helper module for expansion part of U-Net applying either bilinear upsamling and conv or transposed convolution.
    For the bilinear upsamling method, the feature maps stays the same. Just the height and width are doubled.
    Using the transposed convolutions makes it necessary to define the in_ and out_channels because we want to
    halve the feature map and double the height and width using the stride parameter.
    Note that the in_ and out_ channels are divided by 2 because in the upconvolution the tensors from copy and
    current upconvolution part are concatenated.
    """
    def __init__(self, in_channels, out_channels, padding=1, bn=True, upsampling=True):
        super().__init__()
        if upsampling:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, stride=2, kernel_size=2)

        self.doubleconv = DoubleConv(in_channels, out_channels, padding, bn)

    def forward(self, x1, x2):
        """
        UpConvolution operation for the expansion path in the U-Net segmentation model.
        x1 and x2 are concatenated to then perform the doubleconvolution in order to reduce the channel-size of the
        (concatenated) feature maps.
        Look at https://pytorch.org/docs/stable/nn.functional.html#pad for N-dimensional padding
        :param x1 [torch.Tensor]: processed feature_map from the upconvolution-1 which needs to be upsampled
        :param x2 [torch.Tensor]: copy from the feature_map from the downconvolution
        :return:
        """
        x1 = self.up(x1)
        size_upsampled = x1.size() # Batch x Channel x Height x Width
        size_copy = x2.size() # Batch x Channel x Height x Width
        # Now pad x1 with zeros, such that height (h) and width (h) are the same because we later concatenate with x2
        diff_H = (size_copy[2] - size_upsampled[2])
        diff_W = (size_copy[3] - size_upsampled[3])
        padding_left = diff_W // 2
        padding_right = diff_W - (diff_W // 2)
        padding_top  = diff_H // 2
        padding_bottom = diff_H - (diff_H // 2)
        x1 = F.pad(input=x1, pad=[padding_left, padding_right,
                        padding_top, padding_bottom])
        # Concatenate
        x = torch.cat(tensors=[x2, x1], dim=1)
        # Apply DoubleConv operation for reducing the featuremap size
        return self.doubleconv(x)


class OutConv(nn.Module):
    """
    Helper module for the output convolution layer which returns the segmentation map.
    The featuremap only reduces its dimensionality without reducing the height and width.
    """
    def __init__(self, in_channels, n_class):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_class, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


