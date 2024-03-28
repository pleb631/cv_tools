import argparse
from mmcv import Config, DictAction
from apis import OnnxModel, BaseDataset, runner

#os.environ["CUDA_VISIBLE_DEVICES"] ="9"
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config",
        type=str,
    )

    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        default={},
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. For example, "
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'",
    )

    ret = parser.parse_args()
    return ret


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = OnnxModel(**cfg.model)
    dataset = BaseDataset(**cfg.dataset)
    base_runner = runner(model=model, dataset=dataset, **cfg.runner)
    for hook_cfg in cfg.hooks:
        base_runner.register_hook_from_cfg(hook_cfg)
    base_runner.run()



if __name__ == "__main__":
    main()
