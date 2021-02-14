# dataset settings
dataset_type = 'PathologyDataset'
data_root = 'data/gleason_2019/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', imdecode_backend='cv2'),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.9,1.1)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='PathologyDataset',
        data_root=data_root,
        img_dir='train/3-labels/images',
        img_suffix='.png',
        ann_dir='train/3-labels/annotations',
        pipeline=train_pipeline,
        use_patch=False,
        random_sampling=False,
        classes=['benign', 'gleason grade 3', 'gleason grade 4', 'gleason grade 5']),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='valid/images',
        img_suffix='.png',
        ann_dir='valid/annotations',
        pipeline=test_pipeline,
        use_patch=False,
        random_sampling=False,
        classes=['benign', 'gleason grade 3', 'gleason grade 4', 'gleason grade 5']),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='valid/images',
        img_suffix='.png',
        ann_dir='valid/annotations',
        pipeline=test_pipeline,
        use_patch=False,
        random_sampling=False,
        classes=['benign', 'gleason grade 3', 'gleason grade 4', 'gleason grade 5']))
