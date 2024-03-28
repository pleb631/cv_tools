import torch
import cv2
import onnxruntime
import numpy as np
import torchvision
import math


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # Scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    h, w = img.shape[:2]
    s = (int(w * ratio),int(h * ratio))  # new size
    img = cv2.resize(img, s,)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return np.pad(img, [(0, h - s[0]),(0, w - s[1]),(0,0) ], constant_values=0.447*255)  # value = imagenet mean


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
        
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)




def xywh2xyxy(x):
    """
    > It converts the bounding box from x,y,w,h to x1,y1,x2,y2 wherexy1=top-left,xy2=bottom-right
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


class OnnxModel(object):
    def __init__(self, args):
        super(OnnxModel, self).__init__()
        self.img_size = tuple(args.image_size)[::-1]
        self.padding = args.padding
        self.mean = args.mean
        self.std = args.std
        self.args = args
        self.classes = args.classes
        self.obj_threshold = args.obj_threshold
        self.nms_threshold = args.nms_threshold
        self.stride = 32
        self.model = onnxruntime.InferenceSession(str(args.model_path), providers=onnxruntime.get_available_providers())
        self.post_process = self.yolov8PostProcess if args.use_yolov8 else self.yolov5PostProcess
        
    def _forward(self,image):
        
        image = image / 255
        image = np.array((image - self.mean) / self.std, dtype=np.float32)
        image = image.transpose(2, 0, 1)[np.newaxis, :, :, :]
        image = np.ascontiguousarray(image)

        ort_inputs = {self.model.get_inputs()[0].name: image}
        ort_outs = self.model.run(None, ort_inputs)
        boxes = ort_outs[0].squeeze()
        if self.args.use_yolov8:
            boxes = boxes.T
        
        return boxes   
         
    def detect(self, image_ori):
        image = image_ori[:, :, ::-1].copy()
        self.img_h, self.img_w = image.shape[:2]
        image = letterbox(image, self.img_size, stride=self.stride,auto=False,scaleFill=False)[0]
        if not self.args.aug_test:
            boxes = self._forward(image)
        else:
            boxes = self._forward_augment(image)
        boxes = self.post_process(boxes)
        if len(boxes)>0:
            boxes[:, :4] = scale_boxes(self.img_size, boxes[:, :4], [self.img_h, self.img_w]).round()
            # for box in boxes:
            #     cv2.rectangle(image_ori,[int(box[0]),int(box[1])],[int(box[2]),int(box[3])],[255,0,0],3)
            # cv2.imshow("im",image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        return boxes
    def _forward_augment(self, x):
        img_size = x.shape[:2]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 1, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(np.flip(x,fi) if fi else x, si, True,gs=32)
            yi = self._forward(xi)  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return np.concatenate(y, 0)  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        x, y, wh = (
                p[..., 0:1] / scale,
                p[..., 1:2] / scale,
                p[..., 2:4] / scale,
            )  # de-scale
        if flips == 0:
                y = img_size[0] - y  # de-flip ud
        elif flips == 1:
                x = img_size[1] - x  # de-flip lr
        p = np.concatenate((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = 3  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[0] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:-i, :]  # large
        i = (y[-1].shape[0] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][i:,:]  # small
        return y


    def yolov5PostProcess(self, boxes, conf_thres=0.5, iou_thres=0.3):
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 300000  # maximum number of boxes into torchvision.ops.nms()
        max_det=300
        agnostic = False
        
        
        xc = boxes[:,4] > conf_thres
        boxes = boxes[xc,:]
        if len(boxes)==0:
            return np.empty((0,6))
        
        boxes[:, 5:] *= boxes[:, 4:5]

        box = xywh2xyxy(boxes[:, :4])
        
        conf, j = np.max(boxes[:, 5:],1,keepdims=True),np.argmax(boxes[:, 5:],1,keepdims=True)
        x = np.concatenate((box, conf, j), 1)[conf.reshape(-1) >conf_thres]
        
        if len(boxes)==0:
            return np.empty((0,6))
        # Check shape
        
        x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence and remove excess boxes
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 5]  # boxes (offset by class), scores
        boxes = torch.from_numpy(boxes)
        scores = torch.from_numpy(scores)
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        x =  x[i]
        if len(x.shape)==1:
            x=x.reshape(1,-1)


        return x
    
    def yolov8PostProcess(self, boxes, conf_thres=0.5, iou_thres=0.3):
        
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 300000  # maximum number of boxes into torchvision.ops.nms()
        max_det = 300
        agnostic = False
        
        xc = boxes[:,4:].max(1) > conf_thres
        boxes = boxes[xc,]

        box, clss = boxes[:,:4],boxes[:,4:]
        
        

        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        conf = clss.max(1, keepdims=True)
        j = clss.argmax(1, keepdims=True)
        x = np.concatenate((box, conf, j), 1)[conf.reshape(-1) > conf_thres]
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
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        x =  x[i]
        if len(x.shape)==1:
            x=x.reshape(1,-1)
            
        return x
