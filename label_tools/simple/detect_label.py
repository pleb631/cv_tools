"""图像标注脚本, 生成yolo格式的标注文件"""

import cv2
import glob
import os
import numpy as np
import easydict
import sys
from matplotlib import pyplot as plt
ix, iy = -1, -1



def plt_bbox(img, box, line_thickness=None,label_format="{id}",txt_color=(255,255,255),box_color=[255,0,0]):
    

    if isinstance(box, np.ndarray):
        box = box.tolist()
    
    tl = line_thickness or round(
        0.001 * (img.shape[0] + img.shape[1]) / 2)  # line/font thickness
    tl = max(2,tl)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img,p1,p2,box_color,tl)
    if label_format:
        tf = max(tl - 1, 1)  # font thickness
        sf = tl / 3  # font scale
        
        id = int(box[4])
        label = label_format.format(id=id)
        
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img, p1, p2, box_color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
                img,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                sf,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
    return img


def compute_color_for_labels(label):
    color = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [128, 255, 0],
        [255, 128, 0],
    ]
    return color[label % len(color)]


def save_txt(txt_path, info, mode="w"):
    """保存txt文件

    Args:
        txt_path: str, txt文件路径
        info: list, txt文件内容
        mode: str, 'w'代表覆盖写；'a'代表追加写
    """
    os.makedirs(os.path.split(txt_path)[0], exist_ok=True)

    txt_file = open(txt_path, mode)
    for line in info:
        txt_file.write(line + "\n")
    txt_file.close()


def read_txt(txt_path):
    """读取txt文件

    Args:
        txt_path: str, txt文件路径

    Returns:
        txt_data: list, txt文件内容
    """
    txt_file = open(txt_path, "r")
    txt_data = []
    for line in txt_file.readlines():
        txt_data.append(line.replace("\n", ""))

    return txt_data


# 目标框标注程序
class CLabeled:
    def __init__(self, args):

        self.image_folder = args.image_folder

        self.total_image_number = 0
        
        self.images_list = sorted(
            glob.glob(f"{self.image_folder}/**/*.jpg", recursive=True)
            + glob.glob(f"{self.image_folder}/**/*.png", recursive=True),
            reverse=False,
        )
        self._compute_total_image_number()
        
        self.checkpoint_path = os.path.join(args.image_folder, f"checkpoint")
        
        self.current_label_index = 0
        if os.path.exists(self.checkpoint_path):
            self.read_checkpoint(self.checkpoint_path)
            
        self.image = None
        self.current_image = None
        self.label_path = None
        


        self.boxes = list()
        self.classes = list()
        self.cls_num = args.category_num
        self.cur_class = 0

        self.width = None
        self.height = None

        self.windows_name = "image"
        self.mouse_position = (0, 0)
        self.show_label = True
        

    def _reset(self):
        self.image = None
        self.current_image = None
        self.label_path = None
        self.boxes.clear()
        self.classes.clear()
        self.show_label = True

    def _compute_total_image_number(self):
        self.total_image_number = len(self.images_list)


    def _backward(self):
        self.current_label_index -= 1
        self.current_label_index = max(0, self.current_label_index)

    def _roi_limit(self, x, y):
        x, y = min(max(x,0),self.width),min(max(y,0),self.height)
        return x, y

    def change_box_category(self, num=1):

        if len(self.boxes):
            if len(self.boxes) > 1:
                current_point = np.array(
                    [
                        self.mouse_position[0] / self.width,
                        self.mouse_position[1] / self.height,
                    ]
                )
                current_center_point = (
                    np.array([box[0:2] for box in self.boxes])
                    + np.array([box[2:4] for box in self.boxes])
                ) / 2  # 中心点
                square1 = np.sum(np.square(current_center_point), axis=1)
                square2 = np.sum(np.square(current_point), axis=0)
                squared_dist = (
                    -2 * np.matmul(current_center_point, current_point.T)
                    + square1
                    + square2
                )
                sort_index = np.argsort(squared_dist)[0]
            else:
                sort_index = -1

            if 0 <= self.classes[sort_index] + num < self.cls_num:
                self.cur_class = int(self.classes[sort_index] + num)
            elif self.classes[sort_index] + num >= self.cls_num:
                self.cur_class = 0
            else:
                self.cur_class = self.cls_num - 1

            self.classes[sort_index] = self.cur_class

    def box_fix(self, xyxy):
        x_center = float(xyxy[0] + xyxy[2]) / 2
        y_center = float(xyxy[1] + xyxy[3]) / 2
        width = abs(xyxy[2] - xyxy[0])
        height = abs(xyxy[3] - xyxy[1])
        xywh_center = [x_center, y_center, width, height]
        return xywh_center


    def _draw_roi(self, event, x, y, flags, param, mode=True):
        global ix, iy
        x, y = self._roi_limit(x, y)
        self.mouse_position = (x, y)
        self.current_image = self.copy_image()
        if event == cv2.EVENT_LBUTTONDOWN:  # 按下鼠标左键

            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE and not (
            flags == cv2.EVENT_FLAG_LBUTTON
        ):  # 鼠标移动

            cv2.line(
                self.current_image,
                (x, 0),
                (x, self.height),
                (255, 0, 0),
                1,
                8,
            )
            cv2.line(self.current_image, (0, y), (self.width, y), (255, 0, 0), 1, 8)

        elif event == cv2.EVENT_MOUSEMOVE and (
            flags == cv2.EVENT_FLAG_LBUTTON
        ):  # 按住鼠标左键进行移动
            color = compute_color_for_labels(self.cur_class)
            cv2.rectangle(self.current_image, (ix, iy), (x, y), color, 1)

        elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键松开
            if abs(x - ix) > 3 and abs(y - iy) > 3:

                box = [
                    ix / self.width,
                    iy / self.height,
                    x / self.width,
                    y / self.height,
                ]
                box = [
                    max(min(box[0], box[2]), 0),
                    max(min(box[1], box[3]), 0),
                    min(max(box[0], box[2]), 1),
                    min(max(box[1], box[3]), 1),
                ]
                self.boxes.append(box)
                self.classes.append(self.cur_class)

        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.change_box_category()

        elif event == cv2.EVENT_RBUTTONDOWN:  # 删除(中心点或左上点)距离当前鼠标最近的框

            self.current_image = self.copy_image()
            if len(self.boxes):
                if len(self.boxes) > 1:
                    current_point = np.array([x / self.width, y / self.height])
                    current_center_point = (
                        np.array([box[0:2] for box in self.boxes])
                        + np.array([box[2:4] for box in self.boxes])
                    ) / 2  # 中心点
                    square1 = np.sum(np.square(current_center_point), axis=1)
                    square2 = np.sum(np.square(current_point), axis=0)
                    squared_dist = (
                        -2 * np.matmul(current_center_point, current_point.T)
                        + square1
                        + square2
                    )
                    sort_index = np.argsort(squared_dist)[0]

                else:
                    sort_index = -1
                del self.boxes[sort_index]
                del self.classes[sort_index]
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.show_label = not self.show_label

        if self.show_label:
            self._draw_box_on_image(
                self.current_image,
            )
        else:
            cv2.imshow(self.windows_name, self.current_image)

    def _draw_box_on_image(self, image=None,show=True):
        boxes, classes = self.boxes, self.classes
        if image is None:
            image = self.current_image
        for box, cls in zip(boxes, classes):
            x1, y1 = (int(image.shape[1] * box[0]), int(image.shape[0] * box[1]))
            x2, y2 = (int(image.shape[1] * box[2]), int(image.shape[0] * box[3]))
            color = compute_color_for_labels(int(cls))
            box = [x1,y1,x2,y2,int(cls)]
            image = plt_bbox(image, box,box_color=color)
        if show:
            cv2.imshow(self.windows_name, image)
        return image

    def xywh2xyxy(self, xywh):
        """[x, y, w, h]转为[xmin, ymin, xmax, ymax]"""
        xmin = xywh[0] - xywh[2] / 2
        ymin = xywh[1] - xywh[3] / 2
        xmax = xywh[0] + xywh[2] / 2
        ymax = xywh[1] + xywh[3] / 2
        xyxy = [xmin, ymin, xmax, ymax]

        return xyxy

    def read_label_file(self, label_file_path):

        boxes = []
        classes = []

        annotation = read_txt(label_file_path)
        print(annotation)
        for bbox in annotation:
            bbox = list(map(float, bbox.split()))
            boxes.append(self.xywh2xyxy(bbox[1:]))
            classes.append(bbox[0])

        self.boxes = boxes
        self.classes = classes

    def write_label_file(self, label_file_path):
        ann_boxes = []
        for box, cls in zip(self.boxes, self.classes):
            box = list(map(str, self.box_fix(box)))
            box.insert(0, str(int(cls)))
            ann_boxes.append(" ".join(box))

        save_txt(label_file_path, ann_boxes)

    def write_checkpoint(self, checkpoint_path):
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        checkpoint_file = open(checkpoint_path, "w")
        checkpoint_file.writelines(str(self.current_label_index))

    def read_checkpoint(self, checkpoint_path):
        checkpoint_file = open(checkpoint_path, "r")
        for line in checkpoint_file.readlines():
            self.current_label_index = int(line.strip())
        checkpoint_file.close()


    def set_contrast(self, value):
        self.contrast = value
        if not self.image is None:
            self.current_image = self.copy_image()
            self._draw_box_on_image(self.current_image)

    def copy_image(self,):
        image = self.image.copy()
        alpha = self.contrast/10+1
        image = cv2.addWeighted(image, alpha, image, 0, 0)
        return image
    
    
    def run(self):
            
        print("需要标注的图片总数为: ", self.total_image_number)
        cv2.namedWindow(self.windows_name, cv2.WINDOW_NORMAL)
        self.contrast = 0
        cv2.createTrackbar('contrast', self.windows_name, self.contrast, 50, self.set_contrast)

        labeled_index, labeled_num, labeled_person = self.current_label_index, 0, 0
        init = True
        while True:
            if self.current_label_index != labeled_index or init:
                if not init:
                    self.write_label_file(self.label_path)

                    labeled_index = self.current_label_index
                    labeled_num += 1
                    labeled_person += len(self.boxes)
                    print(
                        f"已标注图片数: {labeled_num}; 图片总数：{self.total_image_number}; 已标注数: {labeled_person}\n"
                    )

                init = False
                self.current_label_index = min(
                    self.current_label_index, self.total_image_number - 1
                )

                self.write_checkpoint(self.checkpoint_path)
                self._reset()
                self.image = cv2.imdecode(
                    np.fromfile(
                        self.images_list[self.current_label_index], dtype=np.uint8
                    ),
                    1,
                )

                self.current_image = self.copy_image()

                image_path = self.images_list[self.current_label_index]
                sa, sb = f"{os.sep}images{os.sep}", f"labels"
                if sa in image_path:
                    self.label_path = os.path.join(
                        image_path.rsplit(sa, 1)[0],
                        sb,
                        image_path.rsplit(sa, 1)[1].rsplit(".", 1)[0] + ".txt",
                    )
                else:
                    self.label_path = image_path.rsplit(".", 1)[0] + ".txt"

                if os.path.exists(self.label_path):
                    self.read_label_file(self.label_path)


                print(
                    f"图像ID: {self.current_label_index}\n图像地址: {self.images_list[self.current_label_index]}\nlabel地址: {self.label_path}\n"
                )
                self.width = self.image.shape[1]
                self.height = self.image.shape[0]

                self._draw_box_on_image(self.current_image)

            cv2.setMouseCallback(self.windows_name, self._draw_roi)
            key = cv2.waitKey(0)

            if key == ord("q") or key == ord("Q"):
                self.change_box_category(-1)
                self.current_image = self.copy_image()
                self._draw_box_on_image(self.current_image)
                continue
            elif key == ord("e") or key == ord("E"):
                self.change_box_category(1)
                self.current_image = self.copy_image()
                self._draw_box_on_image(self.current_image)
                continue
            elif key == ord("a") or key == ord("A"):  # 后退一张
                self._backward()
            elif key == ord("d") or key == ord("D"):  # 后退一张
                self.current_label_index = min(
                    self.current_label_index + 1, self.total_image_number - 1
                )
            elif key == ord("l") or key == ord("L"):  # 删除当前图
                os.remove(self.images_list[self.current_label_index])
                if os.path.exists(self.label_path):
                    os.remove(self.label_path)
                del self.images_list[self.current_label_index]
                self._compute_total_image_number()
                self.current_label_index = min(
                    self.current_label_index, self.total_image_number - 1
                )
                init = True
                continue
            elif key == ord("m") or key == ord("M"):
                im = self._draw_box_on_image(self.image.copy())
                im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(16,16),dpi=300)
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                plt.margins(0,0)    
                plt.axis('off')
                plt.imshow(im)
                plt.show()
            elif key == 27:  # 退出
                self.write_label_file(self.label_path)
                break


def main():

    image_folder = sys.argv[1]
       
    if not os.path.exists(image_folder):
            print( f"{image_folder} does not exists! please check it !")
            exit(-1)
            

    category_num = 4
    args = easydict.EasyDict(
        {
            "image_folder": image_folder,
            "category_num": category_num,
        }
    )
    _app = CLabeled(args)
    _app.run()


if __name__ == "__main__":
    main()
