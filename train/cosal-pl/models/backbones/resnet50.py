import torch
import torch.nn.functional as F
from torch import nn

from torchvision.models import resnet50
from collections import OrderedDict
from ..builder import BACKBONES
from .base_backbone import BaseBackbone


@BACKBONES.register_module()
class resnet50(BaseBackbone):
    def __init__(self, pretrained=False):
        super().__init__()

        bb_net = list(resnet50(pretrained=True).children())
        bb_convs = OrderedDict(
            {
                'conv1': nn.Sequential(*bb_net[:3]),
                'conv2': bb_net[4],
                'conv3': bb_net[5],
                'conv4': bb_net[6],
                'conv5': bb_net[7],
            }
        )
        self.ics = [2048, 1024, 512, 256, 64]
        self.encoder = nn.Sequential(bb_convs)

    def forward(self, x):
        conv1_2 = self.encoder.conv1(x)
        conv2_2 = self.encoder.conv2(conv1_2)
        conv3_3 = self.encoder.conv3(conv2_2)
        conv4_3 = self.encoder.conv4(conv3_3)
        conv5_3 = self.encoder.conv5(conv4_3)

        return [conv1_2, conv2_2, conv3_3, conv4_3, conv5_3]
