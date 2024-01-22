# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torch.optim.lr_scheduler as lrs
import cv2
from pathlib import Path
import os


from . import builder


def resize(input, target_size=(224, 224)):
    return F.interpolate(
        input, (target_size[0], target_size[1]), mode="bilinear", align_corners=True
    )


class BaseSEG(pl.LightningModule):
    def __init__(self, workdir, backbone, aux_head, neck, head, train_set, **kwargs):
        super().__init__()

        self.backbone = builder.build_Seg_model(backbone)
        aux_head["in_channels"] = self.backbone.ics[::-1]
        neck["in_channels"] = self.backbone.ics
        head["in_channels"] = self.backbone.ics

        self.aux_head = builder.build_head(aux_head)

        self.neck = builder.build_neck(neck)
        self.train_set = train_set
        self.head = builder.build_head(head)

        self.workdir = workdir

    def training_step(self, batch, **kwargs):
        cosal_im = batch["cosal_img"]
        sal_im = batch["sal_img"]
        group_num = batch["group_num"]
        maps = batch["cosal_gt"]
        cosal_batch = cosal_im.shape[0]
        assert sum(group_num) == cosal_im.shape[0], group_num

        if isinstance(sal_im, torch.Tensor):
            img = torch.cat((cosal_im, sal_im), dim=0)
        else:
            img = cosal_im

        feat = self.backbone(img)
        ALL_SISMs = self.aux_head(feat)

        SISMs = ALL_SISMs[:cosal_batch, ...]
        cmprs_feat = self.neck(feat, cosal_batch)
        pred_list, _ = self.head(feat, cmprs_feat, SISMs, maps, group_num)

        cosal_loss = self.head.get_loss(pred_list, batch["cosal_gt"])

        self.log("cosal", cosal_loss, on_step=True, prog_bar=True, logger=True)

        loss = cosal_loss * 0.9

        if isinstance(sal_im, torch.Tensor):
            SISMs_sup = ALL_SISMs[cosal_batch:, ...]
            aux_loss = self.aux_head.get_loss(SISMs_sup, batch["sal_gt"])
            self.log("aux", aux_loss, on_step=True, prog_bar=True, logger=True)
            loss += 0.1 * aux_loss

        self.log("total_loss", loss, on_step=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, *args, **kwargs):
        img = batch["cosal_img"]
        group_num = batch["group_num"]
        cosal_batch = img.shape[0]
        assert sum(group_num) == img.shape[0]

        feat = self.backbone(img)
        SISMs = self.aux_head(feat)
        cmprs_feat = self.neck(feat, cosal_batch)

        pred = self.head.predict(feat, cmprs_feat, SISMs, group_num)

        cosal_loss = self.head.get_loss(pred, batch["cosal_gt"])

        self.log("val_acc", 1 - cosal_loss, on_epoch=True, logger=True, prog_bar=True)
        return 0

    def predict_step(self, batch, *args, **kwargs):
        img = batch["cosal_img"]
        group_num = batch["group_num"]
        paths = batch["path"]
        cosal_batch = img.shape[0]

        assert sum(group_num) == img.shape[0]

        feat = self.backbone(img)
        SISMs = self.aux_head(feat)
        cmprs_feat = self.neck(feat, cosal_batch)
        pred = self.head.predict(feat, cmprs_feat, SISMs, group_num)
        sals = pred.detach().cpu()

        for sal, path in zip(sals, paths):
            im = cv2.imread(path)
            sal = sal.unsqueeze(0)
            sal = resize(sal, im.shape[:2]).squeeze().numpy()
            sal = sal * 255
            sal = sal.astype("uint8")
            # sal = cv2.applyColorMap(sal, cv2.COLORMAP_JET)
            # im[sal>127,0]=255
            save_path = (Path(self.workdir) / "pred").joinpath(*Path(path).parts[-4:])
            save_path.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(save_path).split(".")[0] + ".png", sal)

    def configure_optimizers(self):
        if "weight_decay" in self.train_set:
            weight_decay = self.train_set["weight_decay"]
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.head.parameters(),
                },
                {
                    "params": self.aux_head.parameters(),
                },
                {
                    "params": self.neck.parameters(),
                },
                {"params": self.backbone.parameters(), "lr": 1e-6},
            ],
            lr=self.train_set["lr"],
            weight_decay=weight_decay,
        )

        if "lr_scheduler" not in self.train_set:
            return optimizer
        else:
            if self.train_set["lr_scheduler"] == "step":
                scheduler = lrs.StepLR(
                    optimizer,
                    step_size=self.train_set["step"],
                    gamma=self.train_set["decay_rate"],
                )

            elif self.train_set["lr_scheduler"] == "multistep":
                scheduler = lrs.MultiStepLR(
                    optimizer,
                    milestones=self.train_set["milestones"],
                    gamma=self.train_set["gamma"],
                )
            elif self.train_set["lr_scheduler"] == "cosine":
                scheduler = lrs.CosineAnnealingLR(
                    optimizer,
                    T_max=self.train_set["T_max"],
                    eta_min=self.train_set["min_lr"],
                )
            else:
                raise ValueError("Invalid lr_scheduler type!")
            return [optimizer], [scheduler]
