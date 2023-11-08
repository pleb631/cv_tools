# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import numpy as np
from pathlib import Path
from itertools import chain
import re
import json
import cv2


from ..pipelines import Compose


class BaseDataset:
    def __init__(self, data_root, pipeline, load_gt=True, **kwargs):
        super(BaseDataset, self).__init__()
        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.formats = [".jpg", ".png", ".jpeg"]
        self.data_infos = self.load_images()
        self.kwargs = kwargs
        self.load_gt = load_gt

    def load_images(
        self,
    ):
        data_infos = sorted(
            chain(*[Path(self.data_root).rglob("*" + f) for f in self.formats])
        )
        return data_infos

    def prepare_data(self, idx):
        image_path = str(self.data_infos[idx])
        info = {"image_file": image_path, "data_root": self.data_root}
        results = copy.deepcopy(info)
        info = self.pipeline(results)
        if self.load_gt:
            try:
                gt = self._load_detect_gt(image_path, **self.kwargs)
            except:
                gt =[]
            info["gt"] = gt

        return info

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def _load_detect_gt(self, image_path, **kwargs):
        try:
            cls2num = kwargs["cls2num"]
        except:
            raise f"cls2num not in kwargs"

        if len(re.findall("images", image_path)) > 1:
            anno_path_part = Path(image_path).parts
            for index in range(len(anno_path_part) - 1, -1, -1):
                if anno_path_part[index] == "images":
                    anno_path_part[index] = "annotations"
                    break
            anno_path = os.path.join(anno_path_part)
            print(image_path)
        else:
            anno_path = image_path.replace("images", "annotations")

        anno_path = os.path.splitext(anno_path)[0] + ".json"
        assert os.path.exists(anno_path), anno_path

        with open(anno_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        gt_b = list()
        for clss, v in data["annotation"].items():
            if clss not in cls2num.keys():
                continue
            for box in v:
                box["box"].extend([1, int(cls2num[clss])])
                gt_box = np.array([box["box"]], dtype=np.float32)
                gt_b.append(gt_box)

        if len(gt_b) > 0:
            gt_b = np.concatenate(gt_b, axis=0)
        else:
            gt_b = np.empty((0, 6))
        return gt_b

    def show_image(self, input_img, pred_b, gt_b, obj_threshold=0, **kwargs):
        input_h, input_w, _ = input_img.shape
        if len(pred_b) > 0:
            pred_b = pred_b[pred_b[:, 4] > obj_threshold]
            pred_b = pred_b * np.array([input_w, input_h, input_w, input_h, 1, 1])
        if len(gt_b) > 0:
            gt_b = gt_b * np.array([input_w, input_h, input_w, input_h, 1, 1])
        if len(gt_b) == 0 and len(pred_b) == 0:
            return None

        for b in pred_b:
            cv2.rectangle(
                input_img,
                (int(b[0]), int(b[1])),
                (int(b[2]), int(b[3])),
                (255, 255, 0),
                2,
            )
            cv2.putText(
                input_img,
                f"{int(b[5])}|{(b[4]):.2f}",
                (int(b[0]) + 15, int(b[1]) + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )
        for b in gt_b:
            cv2.rectangle(
                input_img,
                (int(b[0]), int(b[1])),
                (int(b[2]), int(b[3])),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                input_img,
                str(int(b[5])),
                (int(b[2]) - 10, int(b[3]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
        return input_img
