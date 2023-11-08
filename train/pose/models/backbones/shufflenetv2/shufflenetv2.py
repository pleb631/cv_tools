# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from collections import OrderedDict

from ...builder import BACKBONES
from ..base_backbone import BaseBackbone
from .fpn import FPN, SSH


class IDConv2d(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.conv = nn.Conv2d(inchannels, inchannels, kernel_size=1, bias=False)
        self.w = torch.eye(inchannels, inchannels)
        self.w = self.w.reshape(inchannels, inchannels, 1, 1)

    def forward(self, x):
        return self.conv(x)


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(-1, groups, channels_per_group, height * width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(-1, groups * channels_per_group, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp,
                                    inp,
                                    kernel_size=3,
                                    stride=self.stride,
                                    padding=1,
                                    bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp,
                          branch_features,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features,
                                branch_features,
                                kernel_size=3,
                                stride=self.stride,
                                padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features,
                      branch_features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )
        self.conv = IDConv2d(branch_features)

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i,
                         o,
                         kernel_size,
                         stride,
                         padding,
                         bias=bias,
                         groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((self.conv(x1), self.branch2(x2)), dim=1)
            # out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out





class ShuffleNetv2(nn.Module):
    def __init__(self, radio, pretrained=False):
        super().__init__()
        stages_repeats = [4, 8, 4]
        if radio == 0.5:      
            self.stage_out_channels = [24, 48, 96, 192, 1024]
            pretrained_model = "weights/shufflenetv2_x0.5-f707e7126e.pth"
        elif radio == 1.0:
            self.stage_out_channels = [24, 116, 232, 464, 1024]
            pretrained_model = "/home/dml/project/facequality/weights/shufflenetv2_x1-5666bf0f80.pth"
        else:
            raise ValueError("Invalid radio")

        input_channels = 3
        output_channels = self.stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_names = [F"stage{i}" for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self.stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(
                    InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        self.init_params()

        if pretrained:
            _state_dict = self.state_dict()
            ckpt = torch.load(pretrained_model)
            new_state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k in ckpt.keys():
                    shape_1 = v.shape
                    shape_2 = ckpt[k].shape
                    if shape_1 == shape_2:
                        new_state_dict[k] = ckpt[k]
            _state_dict.update(new_state_dict)
            self.load_state_dict(_state_dict)
            print("Load pretrain model successfully")

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        outputs.append(x)
        x = self.stage3(x)
        outputs.append(x)
        x = self.stage4(x)
        outputs.append(x)
        return outputs


@BACKBONES.register_module()
class ShuffleNetv2FPN(BaseBackbone):
    def __init__(self, radio):
        super().__init__()
        self.body = ShuffleNetv2(radio, pretrained=True)
        if radio == 0.5:
            in_channels_list = [48, 96, 192]
        elif radio == 1.0:
            in_channels_list = [116, 232, 464]
        else:
            raise ValueError("Invalid radio")

        self.fpn = FPN(in_channels_list, 64)
        self.ssh1 = SSH(64, 64)
        self.ssh2 = SSH(64, 64)
        self.ssh3 = SSH(64, 64)
        self.avg_pool1 = nn.AvgPool2d(24)
        self.avg_pool2 = nn.AvgPool2d(12)
        self.avg_pool3 = nn.AvgPool2d(6)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        x1 = self.avg_pool(feature1)
        x1 = x1.view(-1, 64)

        feature2 = self.ssh2(fpn[1])
        x2 = self.avg_pool(feature2)
        x2 = x2.view(-1, 64)
        
        feature3 = self.ssh3(fpn[2])
        x3 = self.avg_pool(feature3)
        x3 = x3.view(-1, 64)
        
        x = torch.cat([x1, x2, x3], 1)

        return x

