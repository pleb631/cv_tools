import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tqdm
import cv2


from tools import *


def imread(path) -> np.ndarray:
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    return bgrImage


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_folder",
        type=str,
        default=r"save_path",
        help="path of detect results",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="save_path",
        help="path of saving metrics",
    )
    args = parser.parse_args()
    return args


num2cls = {0: "license"}
num2cls = {0: "backpack", 1: "suitcase"}
num2cls = {0: "persons", 1: "cars"}
cls2num = {num2cls[key]: key for key in num2cls.keys()}


def main(args):
    col = [("score", f"{x:.2f}") for x in np.arange(0, 1, 0.1)]

    row, pr = list(), list()
    for result_file in Path(args.result_folder).rglob("*.json"):
        model_name = result_file.stem
        clss = model_name
        if clss not in cls2num:
            continue
        print("class:", clss)
        print(f"processing model: {model_name} ......")
        row.append((result_file.parent.stem + "_" + model_name, "recall"))
        row.append((result_file.parent.stem + "_" + model_name, "precision"))
        save_path = Path(args.save_folder) / (
            result_file.parent.stem + "_" + model_name
        )
        Path(save_path).mkdir(parents=True, exist_ok=True)
        with open(result_file) as f:
            data = json.load(f)

        tp_conf, rec, prec = list(), list(), list()
        for score_threshod in np.arange(0, 1, 0.1):
            # for score_threshod in [0,0.5]:
            pred_num, gt_num, tp = 0, 0, 0
            for item in tqdm.tqdm(data):
                if len(item["results"]) == 0:
                    pred_boxes = np.ones((0, 6))
                else:
                    pred_boxes = np.array(item["results"])
                    pred_boxes = pred_boxes[pred_boxes[:, 4] >= score_threshod]
                anno_path = (
                    item["image_path"]
                    .replace("/images", "/annotations")
                    .replace(".jpg", ".json")
                    .replace(".png", ".json")
                )
                gt_boxes = load_gt_boxes(
                    anno_path, clss, cls2num[clss]
                )  # 获取每个图像对应的label

                if len(pred_boxes.shape) == 1:
                    pred_boxes = pred_boxes[np.newaxis, :]
                pred_num += pred_boxes.shape[0]
                gt_num += len(gt_boxes)

                single_tp, conf, preds, gts, needdraw = compute_metric(
                    pred_boxes, gt_boxes, iou_threshold=0.5
                )  # 对单个图像的预测结果计算true positive值
                tp += single_tp
                if score_threshod == 0:
                    tp_conf.extend(conf)

            recall = round(tp / (gt_num + 1e-5), 4)
            rec.append(recall)
            precision = round(tp / (pred_num + 1e-5), 4)
            prec.append(precision)
        pr.append(rec)
        pr.append(prec)

        tp_conf = np.array(tp_conf)
        index = np.argsort(-tp_conf[:, 1])
        tp_conf = tp_conf[index]
        tp_conf = np.concatenate((tp_conf, [[0, gt_num]]))
        with open(f"{save_path}/confidence.json", "w", encoding="utf-8") as f:
            json.dump(tp_conf.tolist(), f, indent=4)

    df = pd.DataFrame(
        pr,
        index=pd.MultiIndex.from_tuples(row, names=["model name", "metrics"]),
        columns=pd.MultiIndex.from_tuples(col),
    )
    df.to_excel(f"{args.save_folder}/pr.xlsx")


if __name__ == "__main__":
    main(args())
