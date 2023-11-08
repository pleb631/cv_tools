_base_ = ["./default_runtime.py", "./yewu.py"]
evaluation = dict(interval=1, metric=["NME"], save_best="NME")

optimizer = dict(
    type="Adam",
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200],
)
total_epochs = 210
log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)

channel_cfg = dict(
    num_output_channels=5,
    dataset_joints=5,
    dataset_channel=[
        list(range(5)),
    ],
    inference_channel=list(range(5)),
)

# model settings
model = dict(
    type="TopDown",
    #pretrained="torchvision://resnet50",
    backbone=dict(type="RepVGG",arch="B0",strides=(2, 2, 2, 2)),
    neck=dict(type="GlobalAveragePooling"),
    keypoint_head=dict(
        type="DeepposeRegressionHead",
        in_channels=1280,
        num_joints=channel_cfg["num_output_channels"],
        loss_keypoint=dict(type="WingLoss", use_target_weight=True),
    ),
    train_cfg=dict(),
    test_cfg=dict(flip_test=False),
)

data_cfg = dict(
    image_size=[112, 112],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg["num_output_channels"],
    num_joints=channel_cfg["dataset_joints"],
    dataset_channel=channel_cfg["dataset_channel"],
    inference_channel=channel_cfg["inference_channel"],
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    #dict(type="TopDownGetBboxCenterScale", padding=1.25),
    dict(type="TopDownRandomFlip", flip_prob=0.5),
    #dict(type="TopDownGetRandomScaleRotation", rot_factor=30, scale_factor=0.25),
    dict(type="TopDownAffine"),
    dict(type="ToTensor"),
    dict(type="NormalizeTensor", mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    dict(type="TopDownGenerateTargetRegression"),
    dict(
        type="Collect",
        keys=["img", "target", "target_weight"],
        meta_keys=[
            "image_file",
            "joints_3d",
            "joints_3d_visible",
            "center",
            "scale",
            "rotation",
            "flip_pairs",
        ],
    ),
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    #dict(type="TopDownGetBboxCenterScale", padding=1.25),
    dict(type="TopDownAffine"),
    dict(type="ToTensor"),
    dict(type="NormalizeTensor", mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    dict(
        type="Collect",
        keys=["img"],
        meta_keys=["image_file", "center", "scale", "rotation", "flip_pairs"],
    ),
]

test_pipeline = val_pipeline

data_root = "/mnt/data4/dataset/face_landmark/"
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type="FaceyewuDataset",
        ann_file=f"{data_root}/train/anno.json",
        img_prefix=f"{data_root}/train/",
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
    val=dict(
        type="FaceyewuDataset",
        ann_file=f"{data_root}/val/anno.json",
        img_prefix=f"{data_root}/val/",
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
    test=dict(
        type="FaceyewuDataset",
        ann_file=f"{data_root}/val/anno.json",
        img_prefix=f"{data_root}/val/",
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
)
