import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from evaluator import Eval_thread
from dataloader import EvalDataset


def main(cfg):
    pred_dir = cfg.pred_dir
    if cfg.save_dir is not None:
        output_dir = cfg.save_dir
    else:
        output_dir = pred_dir
    gt_dir = osp.join(cfg.dataset_dir)
    pred_dir = osp.join(cfg.pred_dir)
    if cfg.methods is None:
        method_names = os.listdir(pred_dir)
    else:
        method_names = cfg.methods.split('+')
    if cfg.datasets is None:
        dataset_names = os.listdir(gt_dir)
    else:
        dataset_names = cfg.datasets.split('+')

    threads = []
    for dataset in dataset_names:
        for method in method_names:
            loader = EvalDataset(osp.join(pred_dir, method, 'pred',dataset),
                                 osp.join(gt_dir, dataset,'gt'))
            thread = Eval_thread(loader, method, dataset, output_dir, cfg.cuda)
            threads.append(thread)
    for thread in threads:
        print(thread.run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default=None)
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument('--dataset_dir', type=str, default='./')
    parser.add_argument('--pred_dir', type=str, default='./')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--cuda', type=bool, default=True)
    config = parser.parse_args()
    main(config)
