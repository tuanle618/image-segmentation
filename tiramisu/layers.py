import torch
import torch.nn as nn
import torch.nn.functional as F

### Layer Modules ###

class DenseLayer(nn.Module):
    """
    Helper module for DenseLayer as stated in the paper the DenseLayer consists of:
    [BatchNormalization - ReLU - 3x3Conv - Dropout]
    """
    def __init__(self, in_channels, out_channels, p=0.2):
        super().__init__()
        self.denselayer = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p)
        )
    def forward(self, x):
        return self.denselayer(x)

class TransitionDown(nn.Module):
    """
    Helper module for TransitionDown (TD)  as stated in the paper the TD consists of:
    [BatchNormalization - ReLU - 1x1Conv - Dropout - Maxpool2x2]
    """
    def __init__(self, in_channels, p=0.2):
        super().__init__()
        self.transition_down = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Dropout2d(p),
            nn.MaxPool2d(kernel_size=2)
        )
    def forward(self, x):
        return self.transition_down(x)

class TransitionUp(nn.Module):
    """
    Helper module for TransitionUp (TU) as stated in the paper the TU consists of:
    [3x3TransposedConv with stride 2] to compensate the pooling operation
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transition_up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2)
    def forward(self, x, skip):
        """
        Computes the upsamples tensor using the TU module and concatenates it with the skip connection
        along the channel-dimension
        :param x: Tensor which should be upsampled.
        :param skip: Tensor which will be concatenated with the upsampled tensor.
        :return: Concatenated tensor
        """
        up = self.transition_up(x)
        # crop the
        up = self.center_crop(featuremap=up, max_height=skip.size(2), max_width=skip.size(3))
        #
        x = torch.cat([up, skip], dim=1)
        return x

    @staticmethod
    def center_crop(featuremap, max_height, max_width):
        """
        Returns a center-cropped version of a feature-map wrt. to its spatial resolutions determined by
        maximum height and maximum width
        :param featuremap: tensor feature map of size [batch, channel-dim, height, width]
        :param max_height: maximum height to extract
        :param max_width: maximum width to extract
        :return: center cropped version of the feature-map.
        """
        _, _, h, w = featuremap.size()
        x = (w - max_width) // 2
        y = (h - max_height) // 2
        return featuremap[:, :, y:(y + max_height), x:(x + max_width)]


### Block modules ###
class DenseBlock(nn.Module):
    """
    Implements the DenseBlock containing several denselayers
    """
    def __init__(self, in_channels, growth_rate=12, n_layers=5, upsample=False):
        """
        Initializes a DenseBlock.
        :param in_channels: Dimension of input featuremap
        :param growth_rate: Dimension of outout featuremap for each denselayer. Defaults to 12
        :param n_layers: Number of denselayers in each denseblock. Defaults to 5
        :param upsample: Boolean whether to store the results from the denseblock in a list. Defaults to false
                         because this is only needed for the BottleNeck block
        """
        super().__init__()
        self.upsample = upsample
        # Math behind the in_channels: As we concatenate the output of previous denselayer, the input_channels
        # increase linearly by the growth rate as follows: in_channels_ = in_channels + growth_rate*k_layer
        # where k_layer is the number of current dense_layer.
        # Use nn.ModuleList() since we need to concatenate the outputs from earlier denselayers for concat.
        self.denselayers = nn.ModuleList([DenseLayer(in_channels=in_channels + i*growth_rate,
                                                   out_channels=growth_rate) for i in range(n_layers)])

    def forward(self, x):
        """
        Implements the forward pass of a DenseBlock.
        :param x: Tensor that is processed.
        :return: Processed Tensor through DenseBlock.
        The calculation is as follows: x_l = H_l([x_{l-1}, x_{l-2}, ..., x_0]) where [] means concat to dim=1 and H_l
        is the l-th denselayer.
        """
        if self.upsample:  # Only needed for BottleNeck which outputs the concatenation of the output of denselayers
            new_feature_maps = []
            for l in self.denselayers:
                out = l(x)
                x = torch.cat(tensors=[x, out], dim=1)
                new_feature_maps.append(out)
            return torch.cat(new_feature_maps, dim=1)
        else:
            for l in self.denselayers:
                out = l(x)
                x = torch.cat(tensors=[x, out], dim=1)
            return x

class BottleNeck(nn.Module):
    """
    Helper Module for the BottleNeck block which itself is a DenseBlock but with the difference that the output
    consists only of the concatenation of the denselayer outputs and also not the input.
    """
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.bottleneck = DenseBlock(in_channels, growth_rate, n_layers, upsample=True)

    def forward(self, x):
        return self.bottleneck(x)

"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
db = DenseBlock(in_channels=3).to(device)
x = torch.randn(4, 3, 256, 256, device=device)
db_out = db(x)
print('Shape of input image: {}'.format(x.size()))
print('Shape of output for the first denseblock: {}'.format(db_out.size()))
bn = BottleNeck(in_channels=db_out.size(1), growth_rate=12, n_layers=5).to(device)
bn_out = bn(db_out)
print('Shape of output for the bottleneck: {}'.format(bn_out.size()))
"""
