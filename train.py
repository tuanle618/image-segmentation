from unet.model import UNet
from tiramisu.model import FCDenseNet
from utils.helpers import get_number_params
from utils.transform import my_transforms


unet = UNet()
fcdnet = FCDenseNet()

print('Trainable parameters U-Net: {}'.format(get_number_params(unet)))
print('Trainable parameters FC-DenseNet: {}'.format(get_number_params(fcdnet)))
