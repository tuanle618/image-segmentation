import numpy as np
from skimage.transform import rescale, rotate
from torchvision.transforms import Compose
import torch

def my_transforms(scale=None, angle=None, flip_prob=None):
    transform_list = []
    if scale is not None:
        transform_list.append(Scale(scale))
    if angle is not None:
        transform_list.append(Rotate(angle))
    if flip_prob is not None:
        transform_list.append(HorizontalFlip(flip_prob))
        transform_list.append(VerticalFlip(flip_prob))

    transform_list.append(CustomToTensor())

    return Compose(transform_list)

## Custom Transform Classes:
class Scale(object):
    def __init__(self, scale):
        assert scale > 0.0 and scale < 1.0
        self.scale = scale

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        img_size = image.shape[0]
        # get a new randomized scale because we want to use this transform for data augmentation
        scale = np.random.uniform(low=1.0-self.scale, high=1.0+self.scale)

        # resize the feature image
        image = rescale(
            image, scale=(scale, scale),
            mode='constant',
            multichannel=True,
            anti_aliasing=False
        )

        # resize the feature mask (label)
        mask = rescale(
            mask, scale=(scale, scale),
            mode='constant', multichannel=True,
            preserve_range=True, anti_aliasing=False
        )

        # zero pad the smaller resized image and mask
        if scale < 1.0:
            diff = (img_size - image.shape[0]) / 2.0
            # get padding values for h, w and channel
            padding = ((int(np.floor(diff)), int(np.ceil(diff))),) * 2 + ((0, 0),)
            image = np.pad(image, padding, mode='constant', constant_values=0)
            mask = np.pad(mask, padding, mode='constant', constant_values=0)
        else:
            # retrieve the original size again
            x_min = (image.shape[0]-img_size) // 2
            x_max = x_min + img_size
            image = image[x_min:x_max, x_min:x_max, ...]
            mask = mask[x_min:x_max, x_min:x_max, ...]

        return {'image':image, 'mask':mask}

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = rotate(image, angle, resize=False, preserve_range=True, mode="constant")
        mask = rotate(
            mask, angle, resize=False, order=0, preserve_range=True, mode="constant"
        )
        return {'image':image, 'mask':mask}

class HorizontalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        if np.random.rand() > self.flip_prob:
            return image, mask

        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

        return {'image':image, 'mask':mask}

class VerticalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        if np.random.rand() > self.flip_prob:
            return image, mask

        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()

        return {'image':image, 'mask':mask}

class CustomToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        if np.max(image) > 1:
            image = image/255.0

        image = torch.tensor(data=image, dtype=torch.float32).permute(dims=(2, 0, 1))
        mask = torch.tensor(data=mask, dtype=torch.uint8).permute(dims=(2, 0, 1))

        return {'image':image, 'mask':mask}




