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
import datetime

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(tensor=m.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.zeros_(tensor=m.bias)


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
        "--image_size",
        type=int,
        default=128,
        help="target input image size (default: 128)",
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

    # Init weights for model
    model = model.apply(weights_init)

    transforms = my_transforms(scale=args.aug_scale,
                               angle=args.aug_angle,
                               flip_prob=args.aug_flip)
    print('Trainable parameters for model {}: {}'.format(args.model, get_number_params(model)))


    # create pytorch dataset
    dataset = DataSetfromNumpy(image_npy_path='data/train_img_{}x{}.npy'.format(args.image_size, args.image_size),
                               mask_npy_path='data/train_mask_{}x{}.npy'.format(args.image_size, args.image_size),
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
    writer = SummaryWriter(log_dir=os.path.join(args.logs, args.model))
    optimizer = Adam(params=model.parameters(), lr=args.lr)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5)

    loss_train = []
    loss_valid = []

    # training loop:
    global_step = 0
    for epoch in range(args.epochs):
        eval_count = 0
        epoch_start_time = datetime.datetime.now().replace(microsecond=0)
        # set model into train mode
        model = model.train()
        train_epoch_loss = 0
        valid_epoch_loss = 0
        # tqdm progress bar
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for batch in train_loader:
                # retrieve images and masks and send to pytorch device
                imgs = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device,
                                              dtype=torch.float32 if model.n_classes == 1 else torch.long)

                # compute prediction masks
                predicted_masks = model(imgs)
                if model.n_classes == 1:
                    predicted_masks = torch.sigmoid(predicted_masks)
                elif model.n_classes > 1:
                    predicted_masks = F.softmax(predicted_masks, dim=1)

                # compute dice loss
                loss = dc_loss(y_true=true_masks, y_pred=predicted_masks)
                train_epoch_loss += loss.item()
                # update model network weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # logging
                writer.add_scalar('Loss/train', loss.item(), global_step)
                # update progress bar
                pbar.update(imgs.shape[0])
                # Do evaluation every 25 training steps
                if global_step % 25 == 0:
                    eval_count += 1
                    val_loss = np.mean(eval_net(model, val_loader, device, dc_loss))
                    valid_epoch_loss += val_loss
                    writer.add_scalar('Loss/validation', val_loss, global_step)
                    if model.n_classes > 1:
                        pbar.set_postfix(**{'Training CE loss (batch)': loss.item(),
                                            'Validation CE (val set)': val_loss})
                    else:
                        pbar.set_postfix(**{'Training dice loss (batch)': loss.item(),
                                            'Validation dice loss (val set)': val_loss})

                global_step += 1
                # save images as well as true + predicted masks into writer
                if global_step % args.vis_images == 0:
                    writer.add_images('images', imgs, global_step)
                    if model.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(predicted_masks) > 0.5, global_step)

            # Get estimation of training and validation loss for entire epoch
            valid_epoch_loss /= eval_count
            train_epoch_loss /= len(train_loader)

            # Apply learning rate scheduler per epoch
            scheduler.step(valid_epoch_loss)
            # Only save the model in case the validation metric is best. For the first epoch, directly save
            if epoch > 0:
                best_model_bool = [valid_epoch_loss < l for l in loss_valid]
                best_model_bool = np.all(best_model_bool)
            else:
                best_model_bool = True

            # append
            loss_train.append(train_epoch_loss)
            loss_valid.append(valid_epoch_loss)

            if best_model_bool:
                print('\nSaving model and optimizers at epoch {} with best validation loss of {}'.format(
                    epoch, valid_epoch_loss)
                )
                torch.save(obj={
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict(),
                },
                    f=results_path + '/model_epoch-{}_val_loss-{}.pth'.format(epoch, np.round(valid_epoch_loss, 4))
                )
                epoch_time_difference = datetime.datetime.now().replace(microsecond=0) - epoch_start_time
                print('Epoch: {:3d} time execution: {}'.format(epoch, epoch_time_difference))

    print('Finished training the segmentation model.\nAll results can be found at: {}'.format(results_path))
    # save scalars dictionary as json file
    scalars = {'loss_train': loss_train,
               'loss_valid': loss_valid}
    with open('{}/all_scalars.json'.format(results_path), 'w') as fp:
        json.dump(scalars, fp)

    print('Logging file for tensorboard is stored at {}'.format(args.logs))
    writer.close()


if __name__ == '__main__':
    args = get_argparse()
    main(args)