import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import os


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def create_coco_json(predn, path,_format="coco"):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    
    jdict = []
    if _format=="yolov5":
        path = Path(path)
        image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    elif _format=="coco":
        image_id = path
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": int(p[5]),
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )

    return jdict


def load_detect_gt(
    image_path,
    img_h,
    img_w,
):
    if len(re.findall("images", image_path)) > 1:
        anno_path_part = Path(image_path).parts
        for index in range(len(anno_path_part) - 1, -1, -1):
            if anno_path_part[index] == "images":
                anno_path_part[index] = "labels"
                break
        anno_path = os.path.join(anno_path_part)
        print(image_path)
    else:
        anno_path = image_path.replace("images", "labels")

    anno_path = os.path.splitext(anno_path)[0] + ".txt"
    if not os.path.exists(anno_path):
        return []

    txt_file = open(anno_path, "r", encoding="UTF-8")
    anno = [line.replace("\n", "") for line in txt_file]

    if len(anno) == 0:
        return anno

    gt_b = list()
    for line in anno:
        line = list(map(float, line.split()))
        x1, y1 = line[1] - line[3] / 2, line[2] - line[4] / 2
        x2, y2 = x1 + line[3], y1 + line[4]
        gt_b.append([x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h, 1, int(line[0])])

    gt_b = np.stack(gt_b, axis=0)
    return gt_b


def compute_metric(pred_boxes, gt_boxes):
    classes = list(map(int, set(pred_boxes[:, 5].tolist())))
    tp = defaultdict(lambda: np.zeros(8))

    for clss in classes:
        pred_cls_boxes = pred_boxes[pred_boxes[:, 5] == clss]
        gt_cls_boxes = gt_boxes[gt_boxes[:, 5] == clss]

        if not (len(pred_cls_boxes) and len(gt_cls_boxes)):
            continue

        pred_cls_boxes = pred_cls_boxes[np.argsort(pred_cls_boxes[:, 4])[::-1]]
        for i, iou_thre in enumerate(range(50, 90, 5)):
            gt_matched_flag = [False] * len(gt_cls_boxes)
            for pred_box in pred_cls_boxes:
                overlaps = np.array([iou(pred_box, gt_box) for gt_box in gt_cls_boxes])

                max_overlap = np.max(overlaps)
                max_overlap_index = int(np.argmax(overlaps))

                if (
                    max_overlap > iou_thre / 100
                    and not gt_matched_flag[max_overlap_index]
                ):
                    tp["all"][i] += 1
                    tp[clss][i] += 1
                    gt_matched_flag[max_overlap_index] = True

    return tp


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


def visualize_pred_gt_box(input_img, pred_b, gt_b):
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


