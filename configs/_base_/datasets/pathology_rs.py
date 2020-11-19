# dataset settings
dataset_type = 'PathologyDataset'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='RandomCrop', crop_size=[1024, 1024], cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImagePatch'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='PathologyDataset',
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/annotations',
        pipeline=train_pipeline,
        use_patch=True,
        random_sampling=True,
        classes=['background', 'inflammation', 'low', 'high', 'carcinoma']),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='valid/images',
        ann_dir='valid/annotations',
        pipeline=test_pipeline,
        use_patch=True,
        random_sampling=False,
        horizontal_stride=1024,
        vertical_stride=1024,
        patch_width=1024,
        patch_height=1024,
        classes=['background', 'inflammation', 'low', 'high', 'carcinoma']),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='valid/images',
        ann_dir='valid/annotations',
        pipeline=test_pipeline))
