# !/usr/bin/python3
# coding=utf-8
from itertools import cycle
import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, Sampler
from .pipelines import Compose
import glob
from itertools import chain


def _init_fn(worker_id):
    np.random.seed(666 + worker_id)


def build_file_paths(img_roots):
    file_paths = []
    indices = []
    cur_group_end_index = 0
    for base in img_roots:
        base = os.path.join(base, "img")
        for group_name in os.listdir(base):
            group_path = os.path.join(base, group_name)
            group_file_names = os.listdir(group_path)
            group_file_names.sort()
            cur_group_end_index += len(group_file_names)
            indices.append(cur_group_end_index)
            for file_name in group_file_names:
                file_path = os.path.join(group_path, file_name)
                file_paths.append(file_path)
    return file_paths, indices


class CosalDataset(Dataset):
    def __init__(self, cosal_paths, pipeline, sal_paths=None, **kwargs):
        img_paths, indices = build_file_paths(cosal_paths)
        self.samples = img_paths
        self.indices = indices
        self.len = len(img_paths)
        self.pipeline = Compose(pipeline)

        self.sal_img_len = 0
        if sal_paths:
            sal_img_paths = list(
                chain(
                    *[
                        glob.glob(os.path.join(path, "img/**/*.*"), recursive=True)
                        for path in sal_paths
                    ]
                )
            )
            sal_img_paths += list(
                chain(
                    *[
                        glob.glob(os.path.join(path, "image/**/*.*"), recursive=True)
                        for path in sal_paths
                    ]
                )
            )
            self.samples += sal_img_paths

            self.sal_img_len = len(sal_img_paths)

    def __getitem__(self, idx):
        image_path = self.samples[idx]
        group_path = os.path.dirname(image_path)
        sal = idx >= self.len

        gt_path = (
            image_path.replace("/img", "/gt").replace("/image", "/mask").split(".")[0]
            + ".png"
        )
        assert os.path.exists(gt_path), f"{gt_path} is not exists"
        gt = cv2.imread(gt_path, 0).astype(np.float32)
        image_info = {
            "image_path": image_path,
            "sal": sal,
            "group_path": group_path,
            "gt": gt,
        }
        return self.pipeline(image_info)


class Cosal_Sampler(Sampler):
    def __init__(
        self,
        indices,
        shuffle,
        batch_size,
        group_size,
        sal_batch_size=0,
        sal_len=0,
        **kwargs,
    ):
        self.indices = indices
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.group_size = group_size
        self.sal_batch_size = sal_batch_size
        self.sal_len = sal_len

    def reset_batches_indices(self):
        groups = []
        start_idx = 0
        for end_idx in self.indices:
            group_indices = list(range(start_idx, end_idx))

            # Shuffle "group_indices" if needed.
            if self.shuffle:
                np.random.shuffle(group_indices)

            # Get the size of current image group.
            num = end_idx - start_idx

            idx = 0
            while idx < num:
                group_size = num if self.group_size is None else self.group_size
                group = group_indices[idx : idx + group_size]
                if len(group) < 2:
                    break
                groups.append(group)
                idx += group_size
            start_idx = end_idx

        sal_indices = list(range(self.indices[-1], self.indices[-1] + self.sal_len))

        if self.shuffle:
            np.random.shuffle(groups)
            np.random.shuffle(sal_indices)

        batch = []
        i = 0
        sal_indices = iter(cycle(sal_indices))
        for idx in groups:
            i += 1
            batch.extend(idx)
            if i == self.batch_size:
                i = 0
                if self.sal_batch_size:
                    while i < self.sal_batch_size:
                        sal_idx = next(sal_indices)
                        i += 1
                        batch.append(sal_idx)
                    i = 0
                yield batch
                batch = []

    def __iter__(self):
        return self.reset_batches_indices()

    def collate(self, batch):
        group_num = []
        current = None
        i = 0
        cosal_img = []
        cosal_gt = []
        sal_img = []
        sal_gt = []
        path = []
        entered_sal = False
        for sample in batch:
            if current is None:
                current = sample["group_path"]
            if not sample["sal"]:
                if current == sample["group_path"]:
                    i += 1
                    if self.group_size != None and i >= self.group_size:
                        group_num.append(i)
                        i = 0
                        current = None
                else:
                    group_num.append(i)
                    i = 1
                    current = sample["group_path"]
                cosal_img.append(sample["img"].unsqueeze(0))
                cosal_gt.append(sample["gt"].unsqueeze(0))
            else:
                if not entered_sal and i:
                    group_num.append(i)
                    i = 0
                    entered_sal = True
                sal_img.append(sample["img"].unsqueeze(0))
                sal_gt.append(sample["gt"].unsqueeze(0))
            path.append(sample["image_path"])
        if i > 0:
            group_num.append(i)

        cosal_img = torch.cat(cosal_img, 0)
        cosal_gt = torch.cat(cosal_gt, 0).unsqueeze(1)
        if sal_img:
            sal_img = torch.cat(sal_img, 0)
            sal_gt = torch.cat(sal_gt, 0).unsqueeze(1)

        return {
            "group_num": group_num,
            "path": path,
            "cosal_img": cosal_img,
            "sal_img": sal_img,
            "cosal_gt": cosal_gt,
            "sal_gt": sal_gt,
        }


def build_dataloader(
    cosal_paths: list,
    batch_size,
    group_size=None,
    sal_batch_size=0,
    sal_paths: list = None,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    fix_seed=False,
    **kwargs,
):
    dataset = CosalDataset(
        cosal_paths=cosal_paths,
        sal_paths=sal_paths,
        **kwargs,
    )

    cosal_sampler = Cosal_Sampler(
        indices=dataset.indices,
        sal_len=dataset.sal_img_len,
        shuffle=shuffle,
        batch_size=batch_size,
        group_size=group_size,
        sal_batch_size=sal_batch_size,
    )
    if fix_seed:
        return DataLoader(
            dataset=dataset,
            batch_sampler=cosal_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=cosal_sampler.collate,
            worker_init_fn=_init_fn,
        )

    return DataLoader(
        dataset=dataset,
        batch_sampler=cosal_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=cosal_sampler.collate,
    )


########################### Testing Script ###########################
"""dataset format
dataset1
- img
  - group1
    - *.jpg
    - *.png
    ...
  .. 
- gt
  - group1
    - *.png
    - *.png
    ...
  ..
dataset2
......
dataset3
......

"""
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pprint import pprint

    path = [r"/root/autodl-tmp/Dataset/COCO9213"]
    data_loader = build_dataloader(
        cosal_paths=path,
        sal_paths=path,
        batch_size=2,
        sal_batch_size=10,
        group_size=5,
        num_workers=2,
    )

    for _ in range(2):
        for i in data_loader:
            pprint(i)
            break
