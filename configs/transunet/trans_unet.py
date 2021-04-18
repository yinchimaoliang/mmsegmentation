_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='VisionTransformer',
    config_name='R50-ViT-L_16',
    loss_config=dict(type='FocalLoss', gamma=0.1),
    img_size=512)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
runner = dict(type='IterBasedRunner', max_iters=50000)
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=500, metric='mDice')

# dataset settings
dataset_type = 'PathologyDataset'
data_root = 'data/gleason_2019/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', imdecode_backend='cv2'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.9, 1.1)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=.5, degree_choice=[90, 180, 270]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
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
        img_dir='train/train/images',
        img_suffix='.png',
        ann_dir='train/train/annotations',
        pipeline=train_pipeline,
        use_patch=False,
        random_sampling=False,
        classes=[
            'benign', 'gleason grade 3', 'gleason grade 4', 'gleason grade 5'
        ]),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/valid/images',
        img_suffix='.png',
        ann_dir='train/valid/annotations',
        pipeline=test_pipeline,
        use_patch=False,
        random_sampling=False,
        classes=[
            'benign', 'gleason grade 3', 'gleason grade 4', 'gleason grade 5'
        ]),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='valid/images',
        img_suffix='.png',
        ann_dir='valid/annotations',
        pipeline=test_pipeline,
        use_patch=False,
        random_sampling=False,
        classes=[
            'benign', 'gleason grade 3', 'gleason grade 4', 'gleason grade 5'
        ]))
