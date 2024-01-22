# Copyright (c) OpenMMLab. All rights reserved.
import logging
from abc import ABCMeta, abstractmethod

import torch.nn as nn


import lightning.pytorch as pl


class BaseBackbone(nn.Module):
    @abstractmethod
    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        """

    def training_step(self, batch):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss
