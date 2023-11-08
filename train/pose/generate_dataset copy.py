import shutil
import os
from pathlib import Path
import glob
import json, csv, pickle, yaml
from tqdm import tqdm
import numpy as np
import cv2
import argparse
import time
import random


def save_json(json_path, info, indent=4, mode='w', with_return_char=False):
    '''保存json文件

    Args:
        json_path: str, json文件路径
        info: dict, json文件内容
        indent: int, 缩进量，默认为4；None代表不缩进
        mode: str, 'w'代表覆盖写；'a'代表追加写
        with_return_char: bool, 写文件时是否在结尾添加换行符
    '''
    #os.makedirs(os.path.split(json_path)[0], exist_ok=True)
    
    # 把python字典转换为字符串
    json_str = json.dumps(info, indent=indent)
    if with_return_char:
        json_str += '\n'
    
    with open(json_path, mode,encoding="UTF-8") as json_file:
        json_file.write(json_str)
    
    json_file.close()
    
    

result = []
root = r"/mnt/data4/dataset/face_landmark/train/"
items = Path(root).rglob("images/*.*")
for item in tqdm(items):
    anno_path = item.parent.parent / "annotations" / item.with_suffix(".json").name
    realpath = item.relative_to(root)
    with open(anno_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        landmark = data["landmark"]
    except:
        landmark = data
    if (np.array(landmark) < -0.1).all():
        continue
    im = cv2.imread(str(item))
    h, w, c = im.shape
    anno = {
        "path": str(realpath),
        "wh": [w, h],
        "landmark": landmark,
    }
    result.append(anno)
print(len(result))
save_json(root+'anno.json',result)
