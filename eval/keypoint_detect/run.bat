goto start
参数如下：
--data_path: 待处理图像路径
--save_path: 图片保存路径
--model_path: 模型路径
--image_size: 图像输入大小
--mean: 均值
--std: 方差
:start
call activate py310
chcp 65001
set PYTHONPATH=.
python evaluate.py ^
--data_path data_path ^
--save_path outputs ^
--model_path C:\Users\lzy\Desktop\repvgg-A1_heatmap.onnx ^
--image_size 128 128 ^
--save_badcase ^
--badcase_th 0.8 ^
--pck_th 0.06 ^
--kpt_indexs 1 ^
--save_badcase ^
--heat_maps
pause