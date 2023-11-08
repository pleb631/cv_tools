import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import os
def creatdict(imgpath, persons):
    params = {"image_path": imgpath,
              "results": persons}
    return params



def load_bag_detect_gt(image_path, filter_area,img_h,img_w,cls2num):
    
    if len(re.findall('images',image_path))>1:
        anno_path_part = Path(image_path).parts
        for index in range(len(anno_path_part)-1,-1,-1):
            if anno_path_part[index]=="images":
                anno_path_part[index]="annotations"
                break
        anno_path = os.path.join(anno_path_part)
        print(image_path)
    else:
        anno_path = image_path.replace("images", "annotations")

    anno_path = os.path.splitext(anno_path)[0]+".json"
    assert os.path.exists(anno_path),anno_path
    
    with open(anno_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_b = list()
    for clss, v in data["annotation"].items():
        if clss not in cls2num.keys():
            continue
        for box in v:
            x1, y1, x2, y2 = box["box"]
            w, h = (x2 - x1) * img_w, (y2 - y1) * img_h
            if w < filter_area or h < filter_area * 2:
                continue
            # if "type" in box and box["type"] == 1:
            #    continue
            box["box"].extend([1, int(cls2num[clss])])
            gt_box = np.array([box["box"]], dtype=np.float32)
            gt_b.append(gt_box)

    if len(gt_b) > 0:
        gt_b = np.concatenate(gt_b, axis=0)
    return gt_b


def compute_metric1(pred_boxes, gt_boxes):
    pred_boxes = pred_boxes[np.argsort(pred_boxes[:, 4])[::-1]]
    tp = defaultdict(lambda:np.zeros(8))

    for i, iou_thre in enumerate(range(50, 90, 5)):
        gt_matched_flag = [False] * len(gt_boxes)
        for pred_box in pred_boxes:
            overlaps = np.array([iou(pred_box, gt_box) for gt_box in gt_boxes])

            max_overlap = np.max(overlaps)
            max_overlap_index = int(np.argmax(overlaps))
            gt_class = int(gt_boxes[max_overlap_index, 5])
            if max_overlap > iou_thre / 100 and not gt_matched_flag[max_overlap_index] and int(pred_box[5]) == gt_class:
                tp["all"][i] += 1
                tp[gt_class][i] += 1
                gt_matched_flag[max_overlap_index] = True

    return tp

def compute_metric(pred_boxes, gt_boxes):
    classes = list(map(int,set(pred_boxes[:, 5].tolist())))
    tp = defaultdict(lambda:np.zeros(8))
    
    
    for clss in classes:
        pred_cls_boxes = pred_boxes[pred_boxes[:, 5]==clss]
        gt_cls_boxes = gt_boxes[gt_boxes[:, 5]==clss]
        
        
        if not (len(pred_cls_boxes)  and  len(gt_cls_boxes)):
            continue
        
        
        pred_cls_boxes = pred_cls_boxes[np.argsort(pred_cls_boxes[:, 4])[::-1]]
        for i, iou_thre in enumerate(range(50, 90, 5)):
            gt_matched_flag = [False] * len(gt_cls_boxes)
            for pred_box in pred_boxes:
                overlaps = np.array([iou(pred_box, gt_box) for gt_box in gt_cls_boxes])

                max_overlap = np.max(overlaps)
                max_overlap_index = int(np.argmax(overlaps))

                if max_overlap > iou_thre / 100 and not gt_matched_flag[max_overlap_index]:
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


def visualize_pred_gt_box(input_img, pred_b, gt_b, draw_img=None, crop_area=None):
    input_h, input_w, _ = input_img.shape
    if len(pred_b) > 0:
        pred_b = pred_b * np.array([input_w, input_h, input_w, input_h, 1, 1])
    if len(gt_b) > 0:
        gt_b = gt_b * np.array([input_w, input_h, input_w, input_h, 1, 1])
    if len(gt_b) == 0 and len(pred_b) == 0:
        return None

    for b in pred_b:
        if draw_img is None:
            cv2.rectangle(input_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 0), 2)
            cv2.putText(input_img, f'{int(b[5])}|{(b[4]):.2f}', (int(b[0]) + 15, int(b[1]) + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 2)
        else:
            b[0] += crop_area[0]
            b[1] += crop_area[1]
            b[2] += crop_area[0]
            b[3] += crop_area[1]
            cv2.rectangle(draw_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
            cv2.putText(draw_img, f'{int(b[5])}|{(b[4]):.2f}', (int(b[0]) + 15, int(b[1]) + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

    for b in gt_b:
        if draw_img is None:
            cv2.rectangle(input_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
            cv2.putText(input_img, str(int(b[5])), (int(b[2]) - 10, int(b[3]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)
        else:
            b[0] += crop_area[0]
            b[1] += crop_area[1]
            b[2] += crop_area[0]
            b[3] += crop_area[1]
            cv2.rectangle(draw_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
            cv2.putText(draw_img, str(int(b[5])), (int(b[2]) - 10, int(b[3]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)

    if draw_img is None:
        return input_img
    else:
        return draw_img




def load_gt_boxes(anno_path, cls,c):
    with open(anno_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gt_b = list()

    for box in data["annotation"][cls]:
        box["box"].append(1)
        # if "type" in box and box["type"] == 1:
        #     continue
        if "class" in list(box.keys()):
            if box["class"] == c:
                box["box"].append(box["class"])
            else:
                continue
        else:
            box["box"].append(0)
        gt_box = np.array([box["box"]], dtype=np.float32).clip(0, 1)
        gt_b.append(gt_box)
    if len(gt_b) > 0:
        gt_b = np.concatenate(gt_b, axis=0)
    return gt_b