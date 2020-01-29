from utils.helpers import get_number_params
from utils.transform import my_transforms
from utils.dataset import DataSetfromNumpy
from eval import eval_net
import numpy as np
import torch.nn.functional as F
import torch
import argparse
from utils.helpers import presentParameters
import os
import json
from loss import DiceLoss
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def save_args(args, modelpath):
    args_file = os.path.join(os.path.join(modelpath, "args.json"))
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)

def get_argparse():
    parser = argparse.ArgumentParser(
        description="Training Image Segmentation Model on Data Science Bowl 2018"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="input batch size for training (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )

    parser.add_argument(
        "--val_percent",
        type=float,
        default=0.30,
        help="Validation percentage of original training set. (default 0.3)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0). If cpu should be used, type 'cpu' ",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="number of workers for data loading (default: 0). Problems with windows dataloader",
    )
    parser.add_argument(
        "--vis_images",
        type=int,
        default=200,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--results", type=str, default="results/dbowl18", help="Path where to save the best model and other results.\
        (default: 'results/dbowl18') ."
    )
    parser.add_argument(
        "--model", type=str, default="fcd-net", help="Which model to train. Either 'u-net' or 'fcd-net'. \
         (default: u-net)",
        choices=['u-net', 'fcd-net']
    )
    parser.add_argument(
        "--logs", type=str, default="logs/", help="folder to save logs"
    )
    parser.add_argument(
        "--images", type=str, default="data/data-science-bowl-2018/stage1_train", help="root folder with images"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--aug_scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug_angle",
        type=int,
        default=30,
        help="rotation angle range in degrees for augmentation (default: 30)",
    )
    parser.add_argument(
        "--aug_flip",
        type=float,
        default=0.5,
        help="flip probability for horizontal and vertical flip (default: 0.5)",
    )
    args = parser.parse_args()

    return args

def main(args):
    presentParameters(vars(args))
    results_path = args.results
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    save_args(args, modelpath=results_path)

    device = torch.device(args.device)
    if args.model == 'u-net':
        from unet.model import UNet
        model = UNet(in_channels=3, n_classes=1).to(device)
    elif args.model == 'fcd-net':
        from tiramisu.model import FCDenseNet
        model = FCDenseNet(in_channels=3, n_classes=1).to(device)
    else:
        print('Parsed model argument "{}" invalid. Possible choices are "u-net" or "fcd-net"'.format(args.model))

    transforms = my_transforms(scale=args.aug_scale,
                               angle=args.aug_angle,
                               flip_prob=args.aug_flip)
    print('Trainable parameters for model {}: {}'.format(args.model, get_number_params(model)))


    # create pytorch dataset
    dataset = DataSetfromNumpy(image_npy_path='data/train_img.npy',
                               mask_npy_path='data/train_mask.npy',
                               transform=transforms)

    # create training and validation set
    n_val = int(len(dataset) * args.val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    ## hacky solution: only add CustomToTensor transform in validation
    from utils.transform import CustomToTensor
    val.dataset.transform = CustomToTensor()

    print('Training the model with n_train: {} and n_val: {} images/masks'.format(n_train, n_val))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    dc_loss = DiceLoss()
    best_validation_dc = 0.0
    writer = SummaryWriter(log_dir=args.logs)
    optimizer = Adam(params=model.parameters(), lr=args.lr)

    loss_train = []
    loss_valid = []
    step = 0

    # training loop:
    global_step = 0
    for epoch in range(args.epochs):
        # set model into train mode
        model = model.train()
        epoch_loss = 0
        # tqdm progress bar
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for batch in train_loader:
                # retrieve images and masks and send to pytorch device
                imgs = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device,
                                              dtype=torch.float32 if model.n_classes == 1 else torch.long)

                # compute prediction masks
                predicted_masks = F.softmax(model(imgs), dim=1)
                # compute dice loss
                loss = dc_loss(y_true=true_masks, y_pred=predicted_masks)
                epoch_loss += loss.item()
                # update model network weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update progress bar per batch-size
                pbar.update(imgs.shape[0])
                global_step += 1

                # do evaluation and logging
                val_loss = eval_net(model, val_loader, device, dc_loss)
                # logging
                writer.add_scalar('Loss/train', loss.item(), global_step)
                if model.n_classes > 1:
                    pbar.set_postfix(**{'Training CE loss (batch)': loss.item(),
                                        'Validation CE': val_loss})
                    writer.add_scalar('Loss/validation', val_loss, global_step)

                else:
                    pbar.set_postfix(**{'Trainig dice loss (batch)': loss.item(),
                                        'Validation dice loss': val_loss})
                    writer.add_scalar('Dice/validation', val_loss, global_step)

                # save images as well as true + predicted masks into writer
                if global_step % args.vis_images == 0:
                    writer.add_images('images', imgs, global_step)
                    if model.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(predicted_masks) > 0.5, global_step)


















if __name__ == '__main__':
    args = get_argparse()
    main(args)
