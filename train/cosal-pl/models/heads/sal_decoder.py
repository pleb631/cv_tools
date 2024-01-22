from torch import nn
import torch.nn.functional as F

from ..builder import HEADS,build_loss
import torch


class LatLayer(nn.Module):
    def __init__(self, in_channel, mid_channel=32):
        super(LatLayer, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.convlayer(x)
        return x


class EnLayer(nn.Module):
    def __init__(self, in_channel=32, mid_channel=32):
        super(EnLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x
    
    
@HEADS.register_module()
class sal_Decoder(nn.Module):
    def __init__(self, in_channels,loss=None,**kwargs):
        super(sal_Decoder, self).__init__()
        
        

        lat_layers = []
        for idx in range(5):
            lat_layers.append(LatLayer(in_channel=in_channels[idx], mid_channel=32))
        self.lat_layers = nn.ModuleList(lat_layers)

        dec_layers = []
        for idx in range(5):
            dec_layers.append(EnLayer(in_channel=32, mid_channel=32))
        self.dec_layers = nn.ModuleList(dec_layers)

        self.top_layer = nn.Sequential(
            nn.Conv2d(in_channels[-1], 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        
        self.loss = build_loss(loss)
        
        
        
    def forward(self, feat_list ):

        feat_top = self.top_layer(feat_list[-1])

        p = feat_top
        for idx in [4, 3, 2, 1, 0]:
            p = self._upsample_add(p, self.lat_layers[idx](feat_list[idx]))
            p = self.dec_layers[idx](p)

        out = self.out_layer(p)
        #out = F.interpolate(out, (224, 224), mode='bilinear', align_corners=True)

        return out
    
    def get_loss(self,sal,gt):
        return self.loss(sal,gt)

    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear') + y
