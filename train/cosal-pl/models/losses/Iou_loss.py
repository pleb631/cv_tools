from torch import nn
import torch.nn.functional as F

from ..builder import LOSSES
import torch


@LOSSES.register_module()
class IoU_loss(nn.Module):
    def __init__(self, **kwargs):
        super(IoU_loss, self).__init__()

    def forward(self, preds, gt):
        if isinstance(preds, list):
            preds = torch.cat(preds, dim=1)

        N, C, H, W = preds.shape
        min_tensor = torch.where(preds < gt, preds, gt)  # shape=[N, C, H, W]
        max_tensor = torch.where(preds > gt, preds, gt)  # shape=[N, C, H, W]
        min_sum = min_tensor.view(N, C, H * W).sum(dim=2)  # shape=[N, C]
        max_sum = max_tensor.view(N, C, H * W).sum(dim=2)  # shape=[N, C]
        loss = 1 - (min_sum / max_sum).mean()
        return loss
