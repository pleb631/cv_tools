# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


from .hook import HOOKS
from .eval import eval

def _compute_box_area(box_a):
    x1, y1, x2, y2 = box_a[:4]

    if x2 > x1 and y2 > y1:
        return (y2 - y1) * (x2 - x1)

    return 0


def _compute_InterIou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 > x1 and y2 > y1:
        intersection = (x2 - x1) * (y2 - y1)
        area_a = _compute_box_area(box_a)
        area_b = _compute_box_area(box_b)
        return intersection/(min(area_a,area_b))
    return 0


def _iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 > x1 and y2 > y1:
        intersection = (x2 - x1) * (y2 - y1)
        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = box_a_area + box_b_area - intersection
        return intersection/(union+1e-10)
    return 0


def process(
    preds,
    gt,
    matches,
    correct_bboxes,
    interiou_threshold=0.8,
    gt_boxes_threshold=[[0, 0], [0, 0]],
    pos_th=0.3,
    **kwargs,
):
    new_match = []
    unmatched_gt = []
    new_gt = []
    correct_bboxes1 = correct_bboxes.copy()
    for match in matches:
        new_gt.append(gt[int(match[0])])
        new_match.append([len(new_gt) - 1, match[1], match[2]])
    for gt_index in range(len(gt)):
        if gt_index not in matches[:, 0]:
            unmatched_gt.append([gt[gt_index], gt_index, int(gt[gt_index][-1])])

    if len(matches) < len(preds) and len(unmatched_gt) > 0:
        # unmatched_gt = np.array(unmatched_gt)
        unmatched_matched_flag = [False] * len(unmatched_gt)
        classes = set(list(zip(*unmatched_gt))[2])
        for clss in classes:
            for pred_index in range(len(preds)):
                if pred_index not in matches[:, 1] or preds[pred_index][5] == clss:
                    overlaps = np.array(
                        [_iou(preds[pred_index], gt_box[0]) for gt_box in unmatched_gt]
                    ) * (unmatched_gt[:5] == clss)

                    max_overlap = np.max(overlaps)
                    max_overlap_index = int(np.argmax(overlaps))
                    if not unmatched_matched_flag[max_overlap_index]:
                        singlegt = unmatched_gt[max_overlap_index][0]
                        InterIou = _compute_InterIou(preds[pred_index], singlegt)
                        if InterIou > interiou_threshold:
                            unmatched_matched_flag[max_overlap_index] = True
                            correct_bboxes1[pred_index][0] = True
                            new_gt.append(unmatched_gt[max_overlap_index][0])
                            new_match.append([len(new_gt) - 1, int(pred_index), 0.495])

    matches1 = np.array(new_match)
    for gt_index in range(len(gt)):
        if not len(new_gt) or (
            len(new_gt) > 0
            and not np.sum(
                np.abs((gt[gt_index][None] - np.array(new_gt))), 1
            ).__contains__(0)
        ):
            if (gt[gt_index][3]) < pos_th:
                fliter_threshold = gt_boxes_threshold[int(gt[gt_index][-1])]
                if fliter_threshold is None or (
                    (gt[gt_index][2] - gt[gt_index][0]) < fliter_threshold[0]
                    and (gt[gt_index][3] - gt[gt_index][1]) < fliter_threshold[1]
                ):
                    continue
            new_gt.append(gt[gt_index])
    gt1 = np.array(new_gt)
    if not len(gt1):
        gt1 = np.empty((0, 6))
    if not len(matches1):
        matches1 = np.empty((0, 3))
    return gt1, matches1, correct_bboxes1


@HOOKS.register_module()
class eval_extra(eval):
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)

    def after_iter(self, runner):
        result = runner.result
        preds = result["pred"]
        gt = result["gt"]
        correct_bboxes = np.zeros((len(preds), 10))
        matches = np.empty((0, 2))
        stat =(correct_bboxes, np.empty(0), np.empty(0), np.empty(0))

        if len(gt) > 0 and len(preds) == 0:
            gt, matches, correct_bboxes = process(
                preds, gt, matches, correct_bboxes, **self.kwargs
            )
            stat=(correct_bboxes, np.empty(0), np.empty(0), gt[:, 5])
        elif len(preds) > 0 and len(gt) > 0:
            correct_bboxes, matches = runner.process_batch(preds, gt)
            if not len(gt) < len(matches):
                gt, matches, correct_bboxes = process(
                    preds, gt, matches, correct_bboxes, **self.kwargs
                )
            stat=(correct_bboxes, preds[:, 4], preds[:, 5], gt[:, 5])
        elif len(preds) > 0 and len(gt) == 0:
            stat=(correct_bboxes, preds[:, 4], preds[:, 5], gt[:, 5])
        self.stats.append(stat)

        runner.result.update(
            {
                "stats": stat,
                "matches": matches,
            }  # array([[label, detect, iou]])
        )

        runner.total_results[result["image_file"]].update(
            {"stats": stat, "matches": matches}
        )
