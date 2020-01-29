import torch.nn as nn
import numpy as np
import torch

class TotalFlatten(nn.Module):
    """
    Module to flatten a feature map of dimension [N, C, H, W] to [N*C*H*W]
    """
    def forward(self, x):
        x = x.view(np.prod(x.size()))
        return x

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0
        self.flatten = TotalFlatten()

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        #flatten the predictions into one array.
        y_pred = self.flatten(y_pred).float()
        y_true = self.flatten(y_true).float()
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

class FocalLoss(nn.Module):

    def __init__(self, alpha=2.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.flatten = TotalFlatten()

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        # flatten the predictions into one array
        y_pred = self.flatten(y_pred).float()
        y_true = self.flatten(y_true).float()
        loss = -(self.alpha*(1-y_pred)**self.gamma * y_true* torch.log(y_pred) +
                 (1-self.alpha) * y_pred**self.gamma * (1-y_true)*torch.log(1-y_pred))
        return loss.mean()


