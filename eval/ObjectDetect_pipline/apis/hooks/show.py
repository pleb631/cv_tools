# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from pathlib import Path


from .hook import HOOKS, Hook


def imread(path) -> np.ndarray:
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    return bgrImage


@HOOKS.register_module()
class show(Hook):
    def __init__(self, save_dir=None, only_save_badcase=False, **kwargs) -> None:
        self.only_save_badcase = only_save_badcase
        self.save_folder = save_dir
        self.kwargs = kwargs

    def before_run(self, runner):
        work_dir = runner.work_dir
        if not self.save_folder:
            self.save_folder = work_dir
        self.save_folder = Path(self.save_folder)

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        result = runner.result
        image = result.get("ori_img", result["image_file"])
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = imread(image)
        pred = result.get("pred", [])
        gt = result.get("gt", [])
        save_img = runner.dataset.show_image(image, pred, gt, **self.kwargs)

        if save_img is not None:
            subpath = Path(result["image_file"]).relative_to(result["data_root"])

            if not self.only_save_badcase:
                save_img_path = self.save_folder / "pred" / subpath
                save_img_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imencode(".jpg", save_img)[1].tofile(save_img_path)

            else:
                metric = runner.total_results[result["image_file"]]["stats"]
                correct_matrix = metric[0]
                conf = metric[1]
                th = self.kwargs.get('obj_threshold',0.5)
                tp = sum(correct_matrix[conf>th][:,0])
                if tp < len(gt):
                    save_img_path = self.save_folder / "pred" / "FN" / subpath
                    save_img_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imencode(".jpg", save_img)[1].tofile(save_img_path)
                if tp < sum(conf>0.5):
                    save_img_path = self.save_folder / "pred" / "FP" / subpath
                    save_img_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imencode(".jpg", save_img)[1].tofile(save_img_path)
