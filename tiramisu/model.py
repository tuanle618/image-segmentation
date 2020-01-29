import torch
import torch.nn as nn
import torch.nn.functional as F
from tiramisu.layers import DenseBlock, BottleNeck, TransitionDown, TransitionUp

class FCDenseNet(nn.Module):
    """
    Implements the Fully-Convolutional DenseNet according to https://arxiv.org/pdf/1611.09326.pdf
    Paper: "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation"
    """
    def __init__(self, in_channels=3, n_classes=1, n_filter_first_conv=48,
                 n_pool=4, growth_rate=12, n_layers_per_block=4, dropout_p=0.2):
        super().__init__()
        """
        This code implements the Fully Convolutional DenseNet described in https://arxiv.org/abs/1611.09326
        The network consist of a downsampling path, where dense blocks and transition down are applied, followed
        by an upsampling path where transition up and dense blocks are applied.
        Skip connections are used between the downsampling path and the upsampling path
        Each layer is a composite function of BN - ReLU - Conv and the last layer is a softmax layer.
        :param in_channels: dimension of image channel. Defaults to 3.
        :param n_classes: number of classes. Defaults to 1.
        :param n_filters_first_conv: number of filters for the first convolution applied. Defaults to 48
        :param n_pool: number of pooling layers = number of transition down = number of transition up. Default 4
        :param growth_rate: number of new feature maps created by each layer in a dense block. Defaults to 12
        :param n_layers_per_block: Integer number of layers per block or a list of size n_pool. Defaults to 4.
        :param dropout_p: dropout rate applied after each convolution (0. for not using). Pytorch handles p=0
        """
        if type(n_layers_per_block) == list:
            assert (len(n_layers_per_block) == 2 * n_pool + 1)
            self.downblocks_list = n_layers_per_block[:n_pool]
            self.upblocks_list = n_layers_per_block[:-n_pool]
            self.bottleneck_list = n_layers_per_block[n_pool+1]
        elif type(n_layers_per_block) == int:
            self.downblocks_list = self.upblocks_list = [n_layers_per_block] * n_pool
            self.bottleneck_list = n_layers_per_block
        else:
            raise ValueError

        self.n_classes = n_classes

        #####################
        # First Convolution #
        #####################
        # Same padding (1), meaning that the output featuremap should have same resolution as the input image
        self.firstConvLayer = nn.Conv2d(in_channels, n_filter_first_conv, kernel_size=3, padding=1)

        #####################
        # Downsampling path #
        #####################

        # Get the current channel dim as variable and iteratively update when instantiating DenseBlocks
        current_filter_dim = n_filter_first_conv  # 48
        # Get the channel dimension for the output of denseblock for later skip connections
        skip_connections_channels_count = []
        self.denseBlocksDown = nn.ModuleList([])  # DenseBlock Module List (DB)
        self.transDownBlocks = nn.ModuleList([])  # TransitionDown Module List (TD) for downsampling.

        for i in range(len(self.downblocks_list)):
            # Dense Block
            self.denseBlocksDown.append(
                DenseBlock(current_filter_dim, growth_rate, self.downblocks_list[i])
            )
            # calculation for current filter dim: current = current + growth_rate*n_layer_per_block
            current_filter_dim += growth_rate*self.downblocks_list[i]
            # insert the current_filter_dim at index 0
            skip_connections_channels_count.insert(0, current_filter_dim)
            # TransitionDown for downsampling
            self.transDownBlocks.append(
                TransitionDown(current_filter_dim, p=dropout_p)
            )

        #####################
        #     Bottleneck    #
        #####################

        # Instantiate the bottleneck which takes the input from the last TD block and applies one DenseBlock
        self.bottleneck = BottleNeck(in_channels=current_filter_dim,
                                     growth_rate=growth_rate, n_layers=self.bottleneck_list)

        previous_block_out = self.bottleneck_list*growth_rate
        current_filter_dim += previous_block_out

        #######################
        #   Upsampling path   #
        #######################

        self.denseBlocksUp = nn.ModuleList([])  # DenseBlock Module List (DB)
        self.transUpBlocks = nn.ModuleList([])  # TransitionUp Module List (TD) for upsampling.

        # Important from paper:
        """
        The upsampled feature maps are then concatenated to the ones coming from 
        the skip connection to form the input of a new dense block.
        Since the upsampling path increases the feature maps spatial
        resolution, the linear growth in the number of features
        would be too memory demanding, especially for the full
        resolution features in the pre-softmax layer.
        In order to overcome this limitation, the input of a dense
        block is not concatenated with its output. Thus, the transposed
        convolution is applied only to the feature maps obtained
        by the LAST DENSE BLOCK and not to all feature maps
        concatenated so far. The last dense block summarizes the
        information contained in all the previous dense blocks at
        the same resolution.
        """
        # Create TransitionUP and DenseBlocks without concatenation
        for i in range(len(self.upblocks_list)-1):
            self.transUpBlocks.append(
                TransitionUp(in_channels=previous_block_out, out_channels=previous_block_out)
            )
            current_filter_dim = previous_block_out + skip_connections_channels_count[i]
            # Do not use entire concatentation but only new feature maps. Upsample=True
            self.denseBlocksUp.append(
                DenseBlock(current_filter_dim, growth_rate, self.upblocks_list[i], upsample=True)
            )
            previous_block_out = growth_rate*self.upblocks_list[i]
            current_filter_dim += previous_block_out

        # Last DenseBlock:
        self.transUpBlocks.append(
            TransitionUp(previous_block_out, previous_block_out)
        )
        current_filter_dim = previous_block_out + skip_connections_channels_count[-1]
        self.denseBlocksUp.append(
            DenseBlock(current_filter_dim, growth_rate, self.upblocks_list[-1], upsample=False)
        )
        current_filter_dim += growth_rate*self.upblocks_list[-1]

        # LastConvolution Layer and Softmax:
        self.lastConvLayer = nn.Conv2d(current_filter_dim, n_classes, kernel_size=1, padding=0)
        # Apply softmax on channel/feature map dimension, i.e. dim=1 in pytorch.
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        """
        Computes the image segmentation map using FCN.
        :param x: input image tensor
        :return: segmentation map of image tensor
        """
        # First Convolution Layer
        x = self.firstConvLayer(x)

        # Apply DownSampling with DenseBlocks and TransitionDown:
        skip_connections = []
        for i in range(len(self.downblocks_list)):
            # Compute DenseBlock outputs which are used as skipconnections in Upsampling
            x = self.denseBlocksDown[i](x)
            # Save skipconnections in list
            skip_connections.append(x)
            # Compute TransitionDown Blocks
            x = self.transDownBlocks[i](x)

        # Apply BottleNeck Block:
        x = self.bottleneck(x)

        # Apply Upsampling with TransitionUp and DenseBlocks:
        for i in range(len(self.upblocks_list)):
            # Use the last element from skip_connections list
            last_skip = skip_connections.pop()
            x = self.transUpBlocks[i](x, last_skip)
            # Apply DenseBlock. Note that TU already returns the concatenation. See layers.TransitionUp class.
            x = self.denseBlocksUp[i](x)

        # Apply last convolution layer
        x = self.lastConvLayer(x)
        # Apply softmax activation
        #x = self.softmax(x)

        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fcn = FCDenseNet(in_channels=3, n_classes=1, n_filter_first_conv=48,
                 n_pool=4, growth_rate=12, n_layers_per_block=4, dropout_p=0.2).to(device)
x = torch.randn(2, 3, 256, 256, device=device)
y = fcn(x)