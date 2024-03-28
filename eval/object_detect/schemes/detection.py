import sys
import cv2
import numpy as np
import os
from collections import defaultdict


sys.path.append("..")
from .load_model import OnnxModel
from uits import *


def imread(path) -> np.ndarray:
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    return bgrImage


class OnnxScheme(object):
    def __init__(self, args):
        super(OnnxScheme, self).__init__()

        self.args = args
        self.model = OnnxModel(args)
        self.classes = args.classes
        
        self.total_results = list()

        self.tp = defaultdict(lambda: np.zeros(2))
        self.num_p = defaultdict(int)
        self.num_g = defaultdict(int)

        self.bais = 1e-9

        self.data_folder = args.data_path
        self.save_folder = args.save_path
        
        self.stats = []

    def run(self, image_path):
        original_image = imread(image_path)
        if len(original_image.shape) == 2:
            original_image = original_image[:, :, np.newaxis].repeat(3, axis=2)
        if original_image.shape[2] > 3:
            original_image = original_image[:, :, :3]
        image = original_image.copy()
        preds = self.model.detect(image)

        if len(preds) and self.args.save_json:
            
            dic0 = create_coco_json(preds,os.path.relpath(image_path,self.data_folder),self.args.save_json)
            self.total_results.extend(dic0)
            
        pred_bbox = list()
        for pred in preds:

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
        if self.args.val or self.args.save_img or self.args.val1:
            gt_b = load_detect_gt(
                str(image_path),
                *original_image.shape[:2],
            )
        if self.args.val:
            if len(gt_b) > 0:
                idx = [
                    i for i in range(len(gt_b)) if gt_b[i, 0] in self.classes
                ]  # 取指定分类
                gt_b = gt_b[idx]
                self.num_g["all"] += gt_b.shape[0]
                for i in self.classes:
                    self.num_g[i] += gt_b[gt_b[:, 0] == i].shape[0]

            self.num_p["all"] += pred_b.shape[0]
            for i in self.classes:
                self.num_p[i] += pred_b[pred_b[:, 5] == i].shape[0]

            if len(pred_b) and len(gt_b):
                tp = compute_metric(pred_b, gt_b)

                self.tp["all"] += tp["all"]
                for i in self.classes:
                    self.tp[i] += tp[i]
        if self.args.val1:
            correct_bboxes = np.zeros((len(pred_b), 10))
            stat = (correct_bboxes, np.empty(0), np.empty(0), np.empty(0))
            if len(pred_b)==0:
                if len(gt_b):
                    stat = (correct_bboxes, np.empty(0), np.empty(0), gt_b[:, 0])
            elif len(gt_b):
                correct_bboxes, _ = process_batch(pred_b, gt_b)
                stat = (correct_bboxes, preds[:, 4], preds[:, 5], gt_b[:, 0])
                
            self.stats.append(stat)

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
            for i, thre in enumerate([50,75]):
                recall[thre] = round(v[i] / (self.num_g[k] + self.bais), 4)
                precision[thre] = round(v[i] / (self.num_p[k] + self.bais), 4)
            print(f"{k} recall\n{recall}")
            print(f"{k} precision\n{precision}")
            file.write(f"{k} recall\n{recall}\n")
            file.write(f"{k} precision\n{precision}\n")
            break
        file.close()
        print("-" * 10)
        
        metrics = DetMetrics(names=self.classes, plot=False)
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]
        if len(stats) and stats[0].any():
            metrics.process(*stats)
        result = metrics.results_dict
        from prettytable import PrettyTable
        tb = PrettyTable()
        tb.field_names = ["precision","recall","mAP50","mAP50-95","fitness","th"]
        tb.add_row([round(i,4) for i in list(result.values())+[float(metrics.th)]])
        print(tb)
        
        if self.args.save_json:
            os.makedirs(self.args.save_path, exist_ok=True)
            json_output = os.path.join(self.args.save_path,  "prediction.json")
            json.dump(self.total_results, open(json_output, "w"), indent=4)
        print("done!!")
