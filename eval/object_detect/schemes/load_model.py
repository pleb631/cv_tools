import torch
import cv2
import onnxruntime
import numpy as np
from torchvision.ops import nms
import os
import copy



def nms_np(dets, thresh=0.3):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]  # xmin
    y1 = dets[:, 1]  # ymin
    x2 = dets[:, 2]  # xmax
    y2 = dets[:, 3]  # ymax
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # argsort()返回数组值从小到大的索引值
    order = scores.argsort()[::-1]    
    keep = []
    while order.size > 0:  # 还有数据
        print("order:",order)
        i = order[0]
        keep.append(i)
        if order.size==1:break
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交框的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        IOU = inter / (areas[i] + areas[order[1:]] - inter)
     
        left_index = (np.where(IOU <= thresh))[0]
        
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[left_index + 1]
        
    return keep



class OnnxModel(object):
    def __init__(self, args):
        super(OnnxModel, self).__init__()
        self.img_size = tuple(args.image_size)
        self.padding = args.padding
        self.mean = args.mean
        self.std = args.std
        self.args = args
        self.classes = args.classes
        self.obj_threshold = args.obj_threshold
        self.nms_threshold = args.nms_threshold
        self.model = onnxruntime.InferenceSession(str(args.model_path), providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        self.post_process = self.yolov8PostProcess if args.use_yolov8 else self.yolov5PostProcess
        
    def single_forward(self,image):
        
        self.img_h, self.img_w = image.shape[:2]
        if self.padding:
            if self.img_w / self.img_h > self.img_size[0] / self.img_size[1]:
                interval_x = 0
                interval_y = int(self.img_w * self.img_size[1] / self.img_size[0] - self.img_h)
            else:
                interval_x = int(self.img_h * self.img_size[0] / self.img_size[1] - self.img_w)
                interval_y = 0
        else:
            interval_x, interval_y = 0, 0
        image = np.pad(image, ((0, interval_y), (0, interval_x), (0, 0)), "constant", constant_values=114)
        new_shape = image.shape[:2][::-1]
        

        image = cv2.resize(image, dsize=self.img_size)
        image = image / 255
        image = np.array((image - self.mean) / self.std, dtype=np.float32)
        image = image.transpose(2, 0, 1)[np.newaxis, :, :, :]
        image = np.ascontiguousarray(image)

        ort_inputs = {self.model.get_inputs()[0].name: image}
        ort_outs = self.model.run(None, ort_inputs)
        boxes = ort_outs[0].squeeze()
        
        boxes = self.post_process(boxes, new_shape, self.obj_threshold, self.nms_threshold)

        return boxes   
         
    def detect(self, image):
        image = image[:, :, ::-1].copy()
        h,w,c=image.shape
        if not self.args.aug_test:
            return self.single_forward(image)
        
        else:
           # self.img_size[0],self.img_size[1]
            h_num=round(h/self.img_size[1])
            w_num = round(w/self.img_size[1])
            h,w,c=image.shape
            single_h = h//h_num
            step_h_pixel  = 3*single_h//5
            step_hnum = (h-single_h)//step_h_pixel+1
            single_w = w//w_num
            step_w_pixel  = 3*single_w//5
            step_wnum = (w-single_w)//step_h_pixel+1
            result = []
            for hi in range(step_hnum):
                for wi in range(step_wnum):
                    x1 = wi*step_w_pixel
                    y1 = hi*step_h_pixel
                    x2 = x1 + single_w if wi<w_num-1 else w
                    y2 = y1+ single_h if hi<h_num-1 else h
                    current_image = image[y1:y2,x1:x2,:].copy()
                    box = self.single_forward(current_image)
                    if len(box)>0:
                        box[:,:4] = box[:,:4]+[x1,y1,x1,y1]
                        result.append(box)
            if len(result)>0:
                x = np.concatenate(result,0)
                c = x[:, 5:6] * 2000 
                boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
                boxes = torch.from_numpy(boxes)
                scores = torch.from_numpy(scores)
                i = nms(boxes, scores, 0.3).tolist()  # NMS
                x = x[i]
                return x.reshape(-1,6)
            else:
                return box
            


    def yolov5PostProcess(self, boxes, new_shape, obj_threshold=0.5, nms_threshold=0.3):
        boxes = boxes.T
        boxes = boxes[boxes[:, 4] > obj_threshold]
        bboxes = np.zeros((boxes.shape[0], 6))
        if boxes.shape[0] > 0:
            bboxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * new_shape[0] / self.img_w
            bboxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * new_shape[1] / self.img_h
            bboxes[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * new_shape[0] / self.img_w
            bboxes[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * new_shape[1] / self.img_h
            bboxes[:, 4] = boxes[:, 4]
            if not True: #是否自带conf
                bboxes[:, 4] = np.max(boxes[:, 4:], axis=1)
                bboxes[:, 5] = np.argmax(boxes[:, 4:], axis=1)
            else:
                if boxes.shape[-1] > 6:
                    bboxes[:, 5] = np.argmax(boxes[:, 5:], axis=1)
                else:
                    bboxes[:, 5] = 0
                
        all_boxes = list()
        for i in self.classes:
            boxes = bboxes[bboxes[:, 5] == i]
            if len(boxes) > 0:
                boxes = torch.from_numpy(boxes)
                boxes = boxes[nms(boxes[:, :4], boxes[:, 4], nms_threshold)]
                boxes[:, 5] = i
                all_boxes.append(boxes)

        if len(all_boxes) > 0:
            all_boxes = torch.cat(all_boxes, dim=0)
            all_boxes[:, :4] = all_boxes[:, :4].clamp(min=0.0, max=1.0)
            all_boxes = all_boxes.numpy()
        return np.array(all_boxes)
    
    def yolov8PostProcess(self, boxes, new_shape, obj_threshold=0.5, nms_threshold=0.3):
        merge = False
        max_wh = 1000  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

        all_boxes = list()


        agnostic = False
        xc = boxes[4:,:].max(0) > obj_threshold
        boxes = boxes[:,xc].T
        #bboxes = np.zeros((boxes.shape[0], 6))
        box, clss = boxes[:,:4],boxes[:,4:]
        
        
        
        def xywh2xyxy(x):
            """
            > It converts the bounding box from x,y,w,h to x1,y1,x2,y2 where xy1=top-left, xy2=bottom-right

            Args:
              x: the input tensor

            Returns:
              the top left and bottom right coordinates of the bounding box.
            """
            y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
            y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
            y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
            y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
            y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
            return y
        
        
    # Apply constraints
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        conf = clss.max(1, keepdims=True)
        j = clss.argmax(1, keepdims=True)
        x = np.concatenate((box, conf, j), 1)[conf.reshape(-1) > obj_threshold]
        # Check shape
        n = x.shape[0]  # number of boxes
        # if not n:  # no boxes
        #     continue
        x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence and remove excess boxes
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        boxes = torch.from_numpy(boxes)
        scores = torch.from_numpy(scores)
        i = nms(boxes, scores, nms_threshold)  # NMS
        #i = i[:max_det]  # limit detections

        all_boxes =  torch.from_numpy(x[i])
        if len(all_boxes.shape)==1:
            all_boxes=all_boxes[None]

        
        all_boxes1= copy.deepcopy(all_boxes)
        all_boxes1[:,0] = all_boxes[:,0]/self.img_size[0]* new_shape[0]
        all_boxes1[:,1] = all_boxes[:,1]/self.img_size[1]* new_shape[1]
        all_boxes1[:,2] = all_boxes[:,2]/self.img_size[0]* new_shape[0]
        all_boxes1[:,3] = all_boxes[:,3]/self.img_size[1]* new_shape[1]

        return np.array(all_boxes1)
