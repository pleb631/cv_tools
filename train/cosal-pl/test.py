import argparse
import os.path as osp
from mmcv import Config, DictAction

from dataset import build_dataloader
from models import BaseSEG
import lightning.pytorch as pl


def parse_args():
    parser = argparse.ArgumentParser(description="cosal")
    parser.add_argument("config", type=str, help="train config file path")
    parser.add_argument("ckpt", type=str, help="train checkpoint file path")
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

    model = BaseSEG(workdir=cfg.workdir, **cfg.model)

    trainer = pl.Trainer()

    # model = model.load_from_checkpoint(args.ckpt)
    datasets = build_dataloader(**cfg.val_data)
    trainer = pl.Trainer(accelerator="gpu")
    trainer.predict(model, dataloaders=datasets, ckpt_path=args.ckpt)

    print("done!!")


if __name__ == "__main__":
    main()
