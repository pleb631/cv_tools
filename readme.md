本仓库为pleb631历史曾经写的代码，为学习框架的副产物，代码具有复用性。

含有以下内容：

```bash
train/
|----- pose 精简的mmpose 2d关键点训练代码
|----- cosal-pl 基于pytorch-lightning的协同分割训练代码
eval/
|----- keypoint_detect 2d关键点模型评估、预测代码
|----- object_detect 基于pycocotools和ultralytics的目标检测评估、预测代码
|----- objectdetect_pipline 可拓展的目标检测流程代码，基于registry机制
label_tools/
|----- keypoint_label 基于pyside6的2d关键点标注工具
|----- simple/
|---------- class_label 基于tk的二分类标注工具
|---------- detect_label 基于opencv-python的二分类目标检测标注工具
```
