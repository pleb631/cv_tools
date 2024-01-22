import argparse
import os.path as osp
from mmcv import Config, DictAction

from dataset import build_dataloader
from models import BaseSEG
import lightning.pytorch as pl
import os


def parse_args():
    parser = argparse.ArgumentParser(description="cosal")
    parser.add_argument("config", type=str, help="train config file path")
    parser.add_argument(
        "--local_log", action="store_true", default=False, help="use csv/yaml logger"
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    datasets = build_dataloader(**cfg.trian_data)
    val_datasets = build_dataloader(**cfg.val_data)
    model = BaseSEG(workdir=cfg.workdir, **cfg.model)
    # ckpt_callback = pl.callbacks.ModelCheckpoint(
    # monitor='loss',
    # save_top_k=1,
    # mode='min'
    # )
    # early_stopping = pl.callbacks.EarlyStopping(monitor = 'val_loss',
    #            patience=3,
    #            mode = 'min')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath=cfg.workdir,
        filename="sample-{epoch:02d}-{val_acc:.2f}",
        every_n_epochs=1,
        mode="max",
    )
    if args.local_log:
        logger = pl.loggers.csv_logs.CSVLogger(
            save_dir=os.path.dirname(cfg.workdir), name=os.path.basename(cfg.workdir)
        )
    else:
        logger = pl.loggers.TensorBoardLogger(
            save_dir=os.path.dirname(cfg.workdir), name=os.path.basename(cfg.workdir)
        )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        max_epochs=70, callbacks=[lr_monitor, checkpoint_callback], logger=logger
    )
    trainer.fit(model=model, train_dataloaders=datasets, val_dataloaders=val_datasets)
    print("done!!")


if __name__ == "__main__":
    main()
