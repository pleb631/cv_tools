import sys
import cv2
import numpy as np
import os
from collections import defaultdict


sys.path.append("..")
from .load_model import OnnxModel
from uits import *


def check_args(args):
    assert args.save_box | args.save_img | args.val, "没有选择模式"


def imread(path) -> np.ndarray:
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    return bgrImage


class OnnxScheme(object):
    def __init__(self, args):
        super(OnnxScheme, self).__init__()

        check_args(args)

        self.args = args
        self.model = OnnxModel(args)
        self.num2cls = args.classes
        self.cls2num = {args.classes[i]: int(i) for i in args.classes.keys()}

        self.classes = list(args.classes.keys())
        self.total_results = defaultdict(list)

        self.tp = defaultdict(lambda: np.zeros(8))
        self.num_p = defaultdict(int)
        self.num_g = defaultdict(int)

        self.bais = 1e-9

        self.data_folder = args.data_path
        self.save_folder = args.save_path
        self.filter_area = args.filter_area

    def run(self, image_path):
        original_image = imread(image_path)
        if len(original_image.shape) == 2:
            original_image = original_image[:, :, np.newaxis].repeat(3, axis=2)
        if original_image.shape[2] > 3:
            original_image = original_image[:, :, :3]
        image = original_image.copy()
        preds = self.model.detect(image, image_path.relative_to(self.data_folder))

        if self.args.save_box:
            ones = defaultdict(list)
            for one in preds.tolist():
                ones[int(one[5])].append(one)

            for key in self.classes:
                dic0 = creatdict(str(image_path), ones[key])
                self.total_results[key].append(dic0)

        pred_bbox = list()
        for pred in preds:
            x1, y1, x2, y2 = pred[:4]
            w = original_image.shape[1] * (x2 - x1)
            h = original_image.shape[0] * (y2 - y1)
            if w < self.filter_area or h < self.filter_area * 2:
                continue
            pred_bbox.append(pred[np.newaxis, :])
        if len(pred_bbox) > 0:
            pred_b = np.concatenate(pred_bbox, axis=0)
            idx = [
                i for i in range(len(pred_b)) if pred_b[i, 5] in self.classes
            ]  # 取指定分类
            pred_b = pred_b[idx]

        else:
            pred_b = np.empty((0, 6))

        gt_b = []
        if self.args.val:
            gt_b = load_bag_detect_gt(
                str(image_path),
                self.filter_area,
                *original_image.shape[:2],
                self.cls2num,
            )

            if len(gt_b) > 0:
                idx = [
                    i for i in range(len(gt_b)) if gt_b[i, 5] in self.classes
                ]  # 取指定分类
                gt_b = gt_b[idx]
                self.num_g["all"] += gt_b.shape[0]
                for i in self.classes:
                    self.num_g[i] += gt_b[gt_b[:, 5] == i].shape[0]

            self.num_p["all"] += pred_b.shape[0]
            for i in self.classes:
                self.num_p[i] += pred_b[pred_b[:, 5] == i].shape[0]

            if len(pred_b) and len(gt_b):
                tp = compute_metric(pred_b, gt_b)

                self.tp["all"] += tp["all"]
                for i in self.classes:
                    self.tp[i] += tp[i]

        if self.args.save_img:
            save_img = visualize_pred_gt_box(image, pred_b, gt_b)

            if save_img is not None:
                subpath = image_path.relative_to(self.data_folder)

                if not self.args.only_save_badcase:
                    save_img_path = self.save_folder / "pred" / subpath
                    save_img_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imencode(".jpg", save_img)[1].tofile(save_img_path)

                else:
                    if tp["all"][0] < len(gt_b):
                        save_img_path = self.save_folder / "pred" / "FN" / subpath
                        save_img_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imencode(".jpg", save_img)[1].tofile(save_img_path)
                    if tp["all"][0] < len(pred_b):
                        save_img_path = self.save_folder / "pred" / "FP" / subpath
                        save_img_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imencode(".jpg", save_img)[1].tofile(save_img_path)

        return 0

    def show_metric(self):
        file_path = self.args.model_path.with_suffix(".txt")
        file = open(file_path, "w+")
        print("computing metrics ......")
        for k, v in self.tp.items():
            recall, precision = dict(), dict()
            for i, thre in enumerate(range(50, 90, 5)):
                recall[thre] = round(v[i] / (self.num_g[k] + self.bais), 4)
                precision[thre] = round(v[i] / (self.num_p[k] + self.bais), 4)
            print(f"{k} recall\n{recall}")
            print(f"{k} precision\n{precision}")
            file.write(f"{k} recall\n{recall}\n")
            file.write(f"{k} precision\n{precision}\n")
        file.close()
        print("-" * 10)
        if self.args.save_box:
            os.makedirs(self.args.save_path, exist_ok=True)
            for key in self.total_results.keys():
                json_output = os.path.join(
                    self.args.save_path, self.num2cls[key] + ".json"
                )  # "backpack","suitcase"
                json.dump(self.total_results[key], open(json_output, "w"), indent=4)
        print("done!!")
