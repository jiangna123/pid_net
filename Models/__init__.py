from __future__ import absolute_import

from Models import deeplabV3_plus
from Models.Aspp_Attunet2 import ASPP_AttUNet2
from Models.Aspp_Pid_Attunet import ASPP_PID_AttUNet

from .Unet import *
from .attunet2 import *
from .ResUnet2 import *

# 通道数
color_mode = {
    1: "grayscale",
    3: "rgb"
}

__model_factory = {
    'unet': unet,
    'attunet': AttUNet,
    'resunet++': ResUnetPlusPlus,
    'deeplabv3+': deeplabV3_plus,
    'aspp-se-att': ASPP_AttUNet2,
    'aspp-att-pid': ASPP_PID_AttUNet,
}


def build_model(name, num_classes, channels, input_height, input_width):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(
            name, avai_models))
    return __model_factory[name](num_classes, channels,
                                 input_height=input_height,
                                 input_width=input_width)
