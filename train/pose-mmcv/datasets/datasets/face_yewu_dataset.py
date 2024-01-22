# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict
import json
import numpy as np
from mmcv import Config, deprecated_api_warning
import cv2

from ..builder import DATASETS
from .kpt_2d_sview_rgb_img_top_down_dataset import Kpt2dSviewRgbImgTopDownDataset


def imread(path) -> np.ndarray:
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    return bgrImage


@DATASETS.register_module()
class FaceyewuDataset(Kpt2dSviewRgbImgTopDownDataset):
    """Face WFLW dataset for top-down face keypoint localization.

    "Look at Boundary: A Boundary-Aware Face Alignment Algorithm",
    CVPR'2018.

    The dataset loads raw images and apply specified transforms
    to return a dict containing the image tensors and other information.

    The landmark annotations follow the 98 points mark-up. The definition
    can be found in `https://wywu.github.io/projects/LAB/WFLW.html`.

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(
        self,
        ann_file,
        img_prefix,
        data_cfg,
        pipeline,
        dataset_info=None,
        test_mode=False,
    ):
        if dataset_info is None:
            warnings.warn(
                "dataset_info is missing. "
                "Check https://github.com/open-mmlab/mmpose/pull/663 "
                "for details.",
                DeprecationWarning,
            )
            cfg = Config.fromfile("configs/_base_/datasets/wflw.py")
            dataset_info = cfg._cfg_dict["dataset_info"]

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode,
            coco_style=False,
        )

        self.ann_info["use_different_joint_weights"] = False
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.anno = data
        self.db = self._get_db()

        print(f"=> num_images: {self.num_images}")
        print(f"=> load {len(self.db)} samples")

    def _get_db(self):
        """Load dataset."""
        gt_db = []
        bbox_id = 0
        num_joints = self.ann_info["num_joints"]
        for anno in self.anno:
            if max(anno["landmark"]) == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
            w, h = anno["wh"]
            keypoints = np.array(anno["landmark"]).reshape(-1, 2)
            joints_3d[:, :2] = keypoints * [w, h]
            joints_3d_visible[:, :2] = keypoints > -0.1
            image_file = anno["path"]
            center = np.array((w // 2, h // 2))
            scale = np.array([w / 200, h / 200])
            bbox = anno.get("bbox", [0,0, w, h])
            # img = imread(image_file)[:,:,::-1]
            gt_db.append(
                {
                    "image_file": image_file,
                    "center": center,
                    "scale": scale,
                    "bbox": bbox,
                    "joints_3d": joints_3d,
                    "joints_3d_visible": joints_3d_visible,
                    "dataset": self.dataset_name,
                    "bbox_id": bbox_id,
                }
            )
            bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x["bbox_id"])
        # if len(gt_db) > 100000:
        #     gt_db = gt_db[:100000]
        self.num_images = len(gt_db)

        return gt_db

    def _get_normalize_factor(self, gts, *args, **kwargs):
        """Get normalize factor for evaluation.

        Args:
            gts (np.ndarray[N, K, 2]): Groundtruth keypoint location.

        Returns:
            np.ndarray[N, 2]: normalized factor
        """

        interocular = np.linalg.norm(gts[:, 0, :] - gts[:, 4, :], axis=1, keepdims=True)
        return np.tile(interocular, [1, 2])

    @deprecated_api_warning(name_dict=dict(outputs="results"))
    def evaluate(self, results, res_folder=None, metric="NME", **kwargs):
        """Evaluate freihand keypoint results. The pose prediction results will
        be saved in ``${res_folder}/result_keypoints.json``.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            results (list[dict]): Testing results containing the following
                items:

                - preds (np.ndarray[1,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - boxes (np.ndarray[1,6]): [center[0], center[1], scale[0], \
                    scale[1],area, score]
                - image_path (list[str]): For example, ['wflw/images/\
                    0--Parade/0_Parade_marchingband_1_1015.jpg']
                - output_heatmap (np.ndarray[N, K, H, W]): model outputs.
            res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
            metric (str | list[str]): Metric to be performed.
                Options: 'NME'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ["NME", "PCK"]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f"metric {metric} is not supported")

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, "result_keypoints.json")
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, "result_keypoints.json")

        kpts = []
        for result in results:
            preds = result["preds"]
            image_paths = result["image_paths"]
            bbox_ids = result["bbox_ids"]

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = image_paths[i][len(self.img_prefix) :]

                kpts.append(
                    {
                        "keypoints": preds[i].tolist(),
                        "image_id": image_id,
                        "bbox_id": bbox_ids[i],
                    }
                )
        kpts = self._sort_and_unique_bboxes(kpts)

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return name_value

    def _sort_and_unique_bboxes(self, kpts, key="bbox_id"):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts
