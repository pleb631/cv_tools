import torch
import onnxruntime
import numpy as np
from torchvision.ops import nms
import copy


class OnnxModel(object):
    def __init__(
        self,
        model_path,
        obj_threshold,
        nms_threshold,
        classes,
        providers,
        yolo5,
        **kwargs,
    ):
        super(OnnxModel, self).__init__()
        self.classes = list(classes.values())
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.model = onnxruntime.InferenceSession(model_path, providers=providers)
        self.post_process = self.yolov5PostProcess if yolo5 else self.yolov8PostProcess

    def detect(self, image_info):
        image = image_info["img"]
        new_shape = image_info["img_shape"]
        image = image.transpose(2, 0, 1)[np.newaxis, :, :, :]
        image = np.ascontiguousarray(image)

        ort_inputs = {self.model.get_inputs()[0].name: image}
        ort_outs = self.model.run(None, ort_inputs)
        boxes = ort_outs[0].squeeze()

        boxes = self.post_process(
             boxes, image_info["Rescale_shape"],new_shape, self.obj_threshold, self.nms_threshold
         )

        return boxes

    def yolov5PostProcess(self, boxes, old_shape,new_shape, obj_threshold=0.5, nms_threshold=0.3):
        img_h,img_w,_ = old_shape
        boxes = boxes[boxes[:, 4] > obj_threshold]
        bboxes = np.zeros((boxes.shape[0], 6))
        if boxes.shape[0] > 0:
            bboxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * new_shape[1] / img_w
            bboxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * new_shape[0] / img_h
            bboxes[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * new_shape[1] / img_w
            bboxes[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * new_shape[0] / img_h
            bboxes[:, 4] = boxes[:, 4]
            if not True:  # 是否自带conf
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
        else:
            all_boxes =np.empty((0,6))
        return np.array(all_boxes)

    def yolov8PostProcess(self, boxes, new_shape, obj_threshold=0.5, nms_threshold=0.3):
        merge = False
        max_wh = 1000  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

        all_boxes = list()

        agnostic = False
        xc = boxes[4:, :].max(0) > obj_threshold
        boxes = boxes[:, xc].T
        # bboxes = np.zeros((boxes.shape[0], 6))
        box, clss = boxes[:, :4], boxes[:, 4:]

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
        x = x[
            x[:, 4].argsort()[::-1][:max_nms]
        ]  # sort by confidence and remove excess boxes
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        boxes = torch.from_numpy(boxes)
        scores = torch.from_numpy(scores)
        i = nms(boxes, scores, nms_threshold)  # NMS
        # i = i[:max_det]  # limit detections

        all_boxes = torch.from_numpy(x[i])
        if len(all_boxes.shape) == 1:
            all_boxes = all_boxes[None]

        all_boxes1 = copy.deepcopy(all_boxes)
        all_boxes1[:, 0] = (
            all_boxes[:, 0] / self.img_size[0] * new_shape[0] / self.img_w
        )
        all_boxes1[:, 1] = (
            all_boxes[:, 1] / self.img_size[1] * new_shape[1] / self.img_h
        )
        all_boxes1[:, 2] = (
            all_boxes[:, 2] / self.img_size[0] * new_shape[0] / self.img_w
        )
        all_boxes1[:, 3] = (
            all_boxes[:, 3] / self.img_size[1] * new_shape[1] / self.img_h
        )

        return np.array(all_boxes1)
