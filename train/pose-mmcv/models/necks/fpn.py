# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
import torch


from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input
    
    
    def init_weights(self):
        pass