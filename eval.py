import torch
import torch.nn.functional as F
from tqdm import tqdm

def eval_net(model, loader, device, loss):
    """
    Evaluation function of segmentation model during training.
    :param model: UNet or FCDenseNet model.
    :param loader: pytorch dataloader to iterate batches over
    :param device: pytorch device
    :param loss: pytorch loss function
    :return:
    """
    # turn model into evaluation mode
    model.eval()
    total_loss = [None] * len(loader)
    for step, batch in enumerate(loader):
        imgs = batch['image'].to(device=device, dtype=torch.float32)
        true_masks = batch['mask'].to(device=device, dtype=torch.float32 if model.n_classes == 1 else torch.long)
        predicted_masks = model(imgs)
        if model.n_classes == 1:
            predicted_masks = torch.sigmoid(predicted_masks)
        elif model.n_classes > 1:
            predicted_masks = F.softmax(predicted_masks, dim=1)
        if model.n_classes > 1:
            total_loss[step] = F.cross_entropy(predicted_masks, true_masks).item()
        else:
            total_loss[step] = loss(predicted_masks, true_masks).item()

    return total_loss

