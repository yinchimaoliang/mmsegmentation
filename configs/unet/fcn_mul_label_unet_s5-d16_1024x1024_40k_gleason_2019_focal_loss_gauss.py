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
        type='FCNMulLabelHead',
        norm_cfg=norm_cfg,
        num_classes=4,
        loss_decode=dict(
             type='FocalLoss', gauss_scale=50, gauss_kernel=5, gauss_sigma=3
        ),
        wei_net_backbone=dict(
            type='ResNet',
            in_channels=3,
            strides=(1, 1, 1, 1),
            depth=18,
            num_stages=4,
            out_indices=(3,),
            style='pytorch'),
        wei_net_conv=dict(
            type='Conv2d',
            in_channels=512,
            out_channels=3,
            kernel_size=3,
            padding=1
        ),
        mul_label_ind=[1, 2, 3],
        final_label_ind=0,
        pretrained='torchvision://resnet18',
        loss_single=dict(
             type='FocalLoss', gauss_scale=50, gauss_kernel=5, gauss_sigma=3
        ),
        sigma=1,
        loss_step=1000
    ),

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
    train=dict(
        ann_dir=['train/train/annotations', 'train/train/Maps1_T', 'train/train/Maps3_T', 'train/train/Maps4_T']),
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