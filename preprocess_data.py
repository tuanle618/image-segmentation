# taken from https://github.com/kamalkraj/DATA-SCIENCE-BOWL-2018/blob/master/data_util.py
# slighty modified with argparse and other os.savings
import os
import random
import sys
import warnings
import numpy as np
from itertools import chain
from skimage.io import imread
from skimage.transform import resize
from skimage.morphology import label
from tqdm import tqdm
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
import argparse

# Setting seed for reproducability
seed = 42
random.seed = seed
np.random.seed = seed

# Data Path
TRAIN_PATH = 'data/data-science-bowl-2018/stage1_train/'
TEST_PATH = 'data/data-science-bowl-2018/stage1_test/'

# Get train and test IDs
train_ids = os.listdir(TRAIN_PATH)
test_ids = os.listdir(TEST_PATH)
# Function read train images and mask return as nump array
def read_train_data(IMG_WIDTH=128, IMG_HEIGHT=128, IMG_CHANNELS=3):
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    if os.path.isfile("data/train_img.npy") and os.path.isfile("data/train_mask.npy"):
        print("Train file loaded from memory")
        X_train = np.load("data/train_img_{}x{}.npy".format(IMG_WIDTH, IMG_HEIGHT))
        Y_train = np.load("data/train_mask_{}x{}.npy".format(IMG_WIDTH, IMG_HEIGHT))
        return X_train, Y_train

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask
    np.save("data/train_img_{}x{}.npy".format(IMG_WIDTH, IMG_HEIGHT), X_train)
    np.save("data/train_mask_{}x{}.npy".format(IMG_WIDTH, IMG_HEIGHT), Y_train)
    return X_train, Y_train


# Function to read test images and return as numpy array
def read_test_data(IMG_WIDTH=128, IMG_HEIGHT=128, IMG_CHANNELS=3):
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    print('\nGetting and resizing test images ... ')
    sys.stdout.flush()
    if os.path.isfile("test_img.npy") and os.path.isfile("test_size.npy"):
        print("Test file loaded from memory")
        X_test = np.load("test_img_{}x{}.npy".format(IMG_WIDTH, IMG_HEIGHT))
        sizes_test = np.load("test_size_{}x{}.npy".format(IMG_WIDTH, IMG_HEIGHT))
        return X_test, sizes_test
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img
    np.save("data/test_img_{}x{}.npy".format(IMG_WIDTH, IMG_HEIGHT), X_test)
    np.save("data/test_size_{}x{}.npy".format(IMG_WIDTH, IMG_HEIGHT), sizes_test)
    return X_test, sizes_test


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


# Iterate over the test IDs and generate run-length encodings for each seperate mask identified by skimage
def mask_to_rle(preds_test_upsampled):
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
    return new_test_ids, rles


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Resizing the images and masks on Data Science Bowl 2018"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Height of resized images. (default: 128)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Width of resized images. (default: 128)",
    )

    args = parser.parse_args()

    print('Creating dataset with spatial dimensions of height: {} and width: {}'.format(args.height, args.width))
    x, y = read_train_data(IMG_HEIGHT=args.height, IMG_WIDTH=args.width)
    x, y = read_test_data(IMG_HEIGHT=args.height, IMG_WIDTH=args.width)