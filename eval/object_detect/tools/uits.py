import json
import numpy as np


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 > x1 and y2 > y1:
        intersection = (x2 - x1) * (y2 - y1)
        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = box_a_area + box_b_area - intersection
        return intersection / union
    return 0


def load_gt_boxes(anno_path, cls,num):
    with open(anno_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gt_b = list()

    for box in data["annotation"][cls]:
        box["box"].append(1)
        # if "type" in box and box["type"] == 1:
        #     continue
        # if "class" in list(box.keys()):
        #     if box["class"] == c:
        #         box["box"].append(box["class"])
        #     else:
        #         continue
        # else:
        box["box"].append(num)
        gt_box = np.array([box["box"]], dtype=np.float32).clip(0, 1)
        gt_b.append(gt_box)
    if len(gt_b) > 0:
        gt_b = np.concatenate(gt_b, axis=0)
    return gt_b



def compute_metric(pred_boxes, gt_boxes, iou_threshold=0.5):
    pred_boxes = pred_boxes[np.argsort(pred_boxes[:, 4])[::-1]]
    tp = 0
    tp_conf = list()
    preds, gts = list(), list()
    gt_matched_flag = [False] * len(gt_boxes)
    needdraw=False
    for pred_box in pred_boxes:
        overlaps = np.array([iou(pred_box, gt_box) for gt_box in gt_boxes])
        if len(overlaps) == 0:
            preds.append(pred_box)
            needdraw=True
            tp_conf.append([0, float(pred_box[4])])
            continue
        max_overlap = np.max(overlaps)
        max_overlap_index = int(np.argmax(overlaps))
        # gt_class = int(gt_boxes[max_overlap_index, 5])
        if max_overlap > iou_threshold and not gt_matched_flag[max_overlap_index] and int(pred_box[5]) == int(gt_boxes[max_overlap_index][5]):
            tp += 1
            gt_matched_flag[max_overlap_index] = True
            tp_conf.append([1, float(pred_box[4])])
        else:
            needdraw=True
            preds.append(pred_box)
            tp_conf.append([0, float(pred_box[4])])
    for i, flag in enumerate(gt_matched_flag):
        if flag == False:
            gts.append(gt_boxes[i])
            tp_conf.append([1, 0])
    return tp, tp_conf, preds, gts, needdraw

