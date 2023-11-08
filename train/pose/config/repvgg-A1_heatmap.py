_base_ = ["./default_runtime.py", "./yewu.py"]

# work_dir='./test/'

evaluation = dict(
    interval=1,
    metric=["PCK", "NME"],
    save_best="PCK",
)

optimizer = dict(
    type="Adam",
    lr=2e-3,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[60, 100]
)
total_epochs = 120
log_config = dict(
    interval=20, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
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
    backbone=dict(type="RepVGG",arch="A1",strides=(2, 2, 2, 2)),
    keypoint_head=dict(
        type="TopdownHeatmapSimpleHead",
        in_channels=1280,
        out_channels=channel_cfg["num_output_channels"],
        # num_deconv_kernels=(4, 4, 4, 4),
        # num_deconv_layers=3,
        # num_deconv_filters=(256, 256, 256),
        loss_keypoint=dict(type="JointsMSELoss", use_target_weight=True),
    ),
)

data_cfg = dict(
    image_size=[128, 128],
    heatmap_size=[32, 32],
    num_output_channels=channel_cfg["num_output_channels"],
    num_joints=channel_cfg["dataset_joints"],
    dataset_channel=channel_cfg["dataset_channel"],
    inference_channel=channel_cfg["inference_channel"],
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type='Albumentation',
        transforms=[
            dict(type="Downscale",scale_min=0.75,scale_max=0.75),
            dict(type='ColorJitter'),
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.3,
                max_width=0.3,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0),
        ]),
    dict(type="TopDownRandomFlip", flip_prob=0.5),
    dict(type="Resize",size=(128,128)),
    dict(type="ToTensor"),
    dict(type="NormalizeTensor", mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    dict(type="TopDownGenerateTarget", sigma=2),
    dict(
        type="Collect",
        keys=["img", "target", "target_weight"],
        meta_keys=[
            "image_file",
            "joints_3d",
            "joints_3d_visible",
            "flip_pairs",
            'center', 'scale','bbox'
        ],
    ),
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize",size=(128,128)),
    dict(type="ToTensor"),
    dict(type="NormalizeTensor", mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    dict(
        type="Collect",
        keys=["img"],
        meta_keys=["image_file", "flip_pairs","bbox_id",'center', 'scale','bbox'],
    ),
]

test_pipeline = val_pipeline

data_root = "/mnt/data4/dataset/face_landmark"
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    train=dict(
        type="FaceyewuDataset",
        ann_file=f"{data_root}/processed_face/anno.json",
        img_prefix=f"{data_root}",
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
