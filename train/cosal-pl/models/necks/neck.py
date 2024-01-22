from torch import nn
import torch.nn.functional as F

from ..builder import NECKS
import torch


@NECKS.register_module()
class neck(nn.Module):
    def __init__(self, in_channels):
        super(neck, self).__init__()

        self.conv6_cmprs = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels[0], 128, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
        )
        self.conv5_cmprs = nn.Conv2d(in_channels[0], 256, 1)
        self.conv4_cmprs = nn.Conv2d(in_channels[1], 256, 1)
        self.conv3_cmprs = nn.Conv2d(in_channels[2], 256, 1)

    def forward(self, feat_list, bs_group):
        conv6_cmprs = self.conv6_cmprs(
            feat_list[-1][:bs_group, ...]
        )  # shape=[N, 128, 7, 7]
        conv5_cmprs = self.conv5_cmprs(
            feat_list[-1][:bs_group, ...]
        )  # shape=[N, 256, 14, 14]
        conv4_cmprs = self.conv4_cmprs(
            feat_list[-2][:bs_group, ...]
        )  # shape=[N, 256, 28, 28]
        conv3_cmprs = self.conv3_cmprs(
            feat_list[-3][:bs_group, ...]
        )  # shape=[N, 128, 56, 56]

        return [conv3_cmprs, conv4_cmprs, conv5_cmprs, conv6_cmprs]
