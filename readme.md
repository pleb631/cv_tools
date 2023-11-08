本仓库j仅提供简单的代码模板，模型评估代码需要根据自己模型的输出格式修改相应的代码。

含有以下内容：

```bash
train/
|----- pose 精简的2d关键点训练代码
eval/
|----- keypoint_detect 2d关键点模型评估、预测代码
|----- object_detect 目标检测评估、预测代码
|----- objectdetect_pipline 可拓展的目标检测流程代码
label_tools/
|----- simple/
|---------- class_label 基于tk的二分类标注工具
|---------- detect_label 基于cv2的二分类目标检测标注工具
```
