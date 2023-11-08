import sys
import cv2
import numpy as np
from pathlib import Path
import os
import json
import cv2

sys.path.append("..")

from utils import *
from KPDModel import KPDModel as OnnxModel


class OnnxScheme(object):
    def __init__(self, args):
        super(OnnxScheme, self).__init__()

        self.args = args
        self.model = OnnxModel(args)
        self.bais = 1e-9

        self.data_folder = args.data_path
        self.save_folder = args.save_path
        self.total_results = list()
        self.result = list()

    def run(self, image_path: str):
        original_image = imread(image_path)
        if len(original_image.shape) == 2:
            original_image = original_image[:, :, np.newaxis].repeat(3, axis=2)
        if original_image.shape[2] > 3:
            original_image = original_image[:, :, :3]
        image = original_image.copy()
        h, w = image.shape[:2]

        if self.args.use_npy:
            npy_root = self.args.npy_root
            npy_path = Path(npy_root) / Path(image_path).relative_to(
                self.data_folder
            ).with_suffix(".npy")
            kpts = np.load(npy_path).transpose()
            kpts = kpts.reshape(-1)[: kpts.size // 2 * 2].reshape(-1, 2)
            pred = np.clip(kpts, 0, 1)
        else:
            pred = self.model.detect(image)
        pred = pred.squeeze()
        if self.args.validated_dataset:
            gt, q = load_kpts_gt(image_path)
            if q == 0:
                return 0
            kpts = gt * [w, h]
            for x, y in kpts.astype(np.int32):
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

            subpath = Path(image_path).relative_to(self.data_folder)

            save_img_path = Path(self.save_folder) / "pred" / subpath
            save_img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".jpg", image)[1].tofile(save_img_path)

            return 0

        if self.args.save_kpts:
            dic = creatdict(image_path, pred, [w, h])
            self.total_results.append(dic)

        if self.args.val:
            gt, q = load_kpts_gt(image_path)

            if np.all(gt < -0.1):
                return 0
            if not len(gt):
                print(f"{image_path} has no anno!!\n")
                return 0
            mask = gt > -0.1
            dic = {
                "pred": pred.reshape(1, -1, 2),
                "gt": gt.reshape(1, -1, 2),
                "mask": mask.reshape(1, -1, 2),
                "image_path": image_path,
                "wh": [w, h],
            }
            self.result.append(dic)

        if self.args.save_pred:
            kpts = pred * [w, h]
            image_copy = image.copy()
            for x, y in kpts.astype(np.int32):
                cv2.circle(image_copy, (x, y), 2, (0, 0, 255), -1)

            subpath = Path(image_path).relative_to(self.data_folder)

            save_img_path = Path(self.save_folder) / "pred" / subpath
            save_img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".jpg", image_copy)[1].tofile(save_img_path)

        if self.args.save_badcase:
            acc, avg_acc, cnt=keypoint_pck_accuracy(
                pred.reshape(1, -1, 2),
                gt.reshape(1, -1, 2),
                mask.reshape(1, -1, 2)[..., 0],
                thr=0.06,
                normalize=np.ones((1, 2), dtype=np.float32),
            )
            if avg_acc<self.args.badcase_th:
                kpts = pred * [w, h]
                image_copy = image.copy()
                for acc_p,(x, y) in zip(acc,kpts.astype(np.int32)):
                    
                    color = (0, 0, 255) if acc_p<self.args.badcase_th else (255, 0, 0)
                    cv2.circle(image_copy, (x, y), 2, color, -1)

                subpath = Path(image_path).relative_to(self.data_folder)

                save_img_path = Path(self.save_folder) / "badcase" / subpath
                save_img_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imencode(".jpg", image_copy)[1].tofile(save_img_path)
                
                
            

        return 0

    def show_metric(self):
        file_path = Path(self.save_folder)/(Path(self.args.model_path).with_suffix(".txt").name)
        os.makedirs(self.save_folder, exist_ok=True)
        if self.args.val:
            print("computing metrics ......")
            with open(file_path, "w+") as f:
                printf(len(self.result))
                for i, method in enumerate(self.args.methods):
                    methodFactory[method](self.result, f, self.args)
                    printf("-" * 10, f)

        if self.args.save_kpts:
            os.makedirs(self.save_folder, exist_ok=True)
            json_output = os.path.join(
                self.save_folder, Path(self.args.model_path).name + ".json"
            )
            with open(json_output, "w") as f:
                json.dump(self.total_results, f, indent=4)
        print("done!!")
