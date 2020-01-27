from unet.model import UNet
from tiramisu.model import FCDenseNet
from utils import get_number_params

unet = UNet()
fcdnet = FCDenseNet()

print('Trainable parameters U-Net: {}'.format(get_number_params(unet)))
print('Trainable parameters FC-DenseNet: {}'.format(get_number_params(fcdnet)))