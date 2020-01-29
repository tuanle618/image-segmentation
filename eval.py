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
    total_loss = 0
    for batch in loader:
        imgs = batch['image'].to(device=device, dtype=torch.float32)
        true_masks = batch['mask'].to(device=device, dtype=torch.float32 if model.n_classes == 1 else torch.long)
        mask_pred = model(imgs)
        for true_mask, pred in zip(true_masks, mask_pred):
            pred = (pred > 0.5).float()
            if model.n_classes > 1:
                total_loss += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
            else:
                total_loss += loss(pred, true_mask).item()


    return total_loss / len(loader)

