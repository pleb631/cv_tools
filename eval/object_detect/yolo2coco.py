import os
import glob
import tqdm
import cv2
import json
import numpy as np

def yolo2coco(root_dir,classes,save_path):
    root_path = root_dir
    print("Loading data from ",root_path)

    assert os.path.exists(root_path)

    # with open(class_path) as f:
    #     classes = f.read().strip().split()
    # images dir name
    image_paths = glob.glob(os.path.join(root_path,"**/*.jpg"),recursive=True)+glob.glob(os.path.join(root_path,"**/*.png"),recursive=True)

    dataset = {'categories': [], 'annotations': [], 'images': []}
    for i, cls in enumerate(classes, 0):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
    
    # 标注的id
    ann_id_cnt = 0
    for k, index in enumerate(tqdm.tqdm(image_paths)):
        # 支持 png jpg 格式的图片。
        txt_path = index.replace('images','labels').replace('.jpg','.txt').replace('.png','.txt')
        # 读取图像的宽和高
        im = cv2.imdecode(np.fromfile(index, np.uint8), cv2.IMREAD_COLOR)
        height, width, _ = im.shape

        # 添加图像的信息
        dataset['images'].append({'file_name': os.path.relpath(index, start=root_path),
                                    'id': os.path.relpath(index, start=root_path),
                                    'width': width,
                                    'height': height})
        
        if not os.path.exists(txt_path):
            # 如没标签，跳过，只保留图片信息。
            continue
        with open(txt_path, 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                # 标签序号从0开始计算, coco2017数据集标号混乱，不管它了。
                cls_id = int(label[0])   
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': os.path.relpath(index, start=root_path),
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    # 保存结果    
    json_name = os.path.join(root_path, 'annotations/{}'.format(save_path))
    
    os.makedirs(os.path.dirname(json_name),exist_ok=True)
    
    with open(json_name, 'w') as f:
        json.dump(dataset, f,ensure_ascii=False)
        print('Save annotation to {}'.format(json_name))


if __name__ == '__main__':
    root_dir = r'D:\project\datasets\coco128'
    class_path = list(range(80))
    save_path =r'gt.json' 
    yolo2coco(root_dir,class_path,save_path)