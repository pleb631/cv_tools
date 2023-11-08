data_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Rescale", size=(256, 128)),  # hw
    dict(type="Pad", size=(256, 128), pad_val=(114, 114, 114)),
    dict(type="Normalize", mean=[0, 0, 0], std=[255, 255, 255], to_rgb=False),
]

classes = dict(backpack=0, suitcase=1)
model = dict(
    model_path=r"best.onnx",
    obj_threshold=0.01,
    nms_threshold=0.3,
    classes=classes,
    providers=["CPUExecutionProvider"],
    yolo5=True,
)

dataset = dict(
    data_root="/temp_data/",
    pipeline=data_pipeline,
    cls2num=classes,
)

work_dir = "./workdir/test"
runner = dict(classes=classes, work_dir=work_dir, overwrite=True)

hooks = [
    dict(
        type="run_model",
        priority=0,
        use_key=None,
        save_key=["pred_box", "gt", "ori_shape"],
        collect_keys=["data_root", "image_file", "gt", "ori_shape"],
    ),
    dict(
        type="eval_extra",
        priority=50,
        interiou_threshold=1,
        gt_boxes_threshold=[[0, 0], [0, 0]],
        pos_th=0,
    ),
    #dict(type="show", priority=100, only_save_badcase=True, obj_threshold=0.5),
]
