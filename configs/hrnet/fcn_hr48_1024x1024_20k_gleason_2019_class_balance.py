_base_ = './fcn_hr18_1024x1024_20k_gleason_2019.py'
norm_cfg = dict(type='BN', requires_grad=True)
data_root = 'data/gleason_2019/'
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        norm_cfg=norm_cfg,
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        type='FCNHead',
        in_channels=[48, 96, 192, 384],
        channels=sum([48, 96, 192, 384]),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
        )
    )
)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU')

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

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='BalanceDataset',
        dataset=dict(
            type='PathologyDataset',
            data_root=data_root,
            img_dir='train/3-labels/images',
            img_suffix='.png',
            ann_dir='train/3-labels/annotations',
            pipeline=train_pipeline,
            use_patch=False,
            random_sampling=False,
            classes=['benign', 'gleason grade 3', 'gleason grade 4', 'gleason grade 5']
        )
    )
)
