_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/gleason_2019.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
dataset_type = 'PathologyDataset'
data_root = 'data/gleason_2019/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(
        norm_cfg=norm_cfg),
    decode_head=dict(
        norm_cfg=norm_cfg,
        num_classes=4,
        loss_decode=dict(
            type='FocalLoss')),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/valid/images',
        img_suffix='.png',
        ann_dir='train/valid/annotations',
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
        classes=['benign', 'gleason grade 3', 'gleason grade 4', 'gleason grade 5'])
    )

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mDice')