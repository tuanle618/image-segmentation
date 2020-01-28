import os
import random
import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset


class DataSetfromNumpy(Dataset):
    """
    Custom dataset object thats loads the entire training images and masks into memory
    """
    def __init__(self,
                 image_npy_path='data/train_img.npy',
                 mask_npy_path='data/train_mask.npy',
                 transform=None):
        self.images = np.load(image_npy_path)
        self.masks = np.load(mask_npy_path)
        self.n = len(self.images)
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        sample = (self.images[item], self.masks[item])

        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample

"""
dataset = DataSetfromNumpy()
img, mask = dataset[1]
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()
plt.imshow(np.squeeze(mask))
plt.show()
"""
