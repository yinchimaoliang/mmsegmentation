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
        type='FCNMulLabelHead',
        in_channels=[48, 96, 192, 384],
        channels=sum([48, 96, 192, 384]),
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
        num_classes=4,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='FocalLoss', gauss_scale=5, gauss_kernel=5, gauss_sigma=3
        ),
        loss_single=dict(
             type='FocalLoss', gauss_scale=5, gauss_kernel=5, gauss_sigma=3
        ),
        sigma=1
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', imdecode_backend='cv2'),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.9,1.1)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=.5, degree_choice=[90, 180, 270]),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]


data = dict(
    train=dict(
        type='BalanceDataset',
        dataset=dict(
            type='PathologyDataset',
            data_root=data_root,
            img_dir='train/3-labels-134/images',
            img_suffix='.png',
            ann_dir=['train/3-labels/annotations', 'train/3-labels-134/Maps1_T', 'train/3-labels-134/Maps3_T', 'train/3-labels-134/Maps4_T'],
            pipeline=train_pipeline,
            use_patch=False,
            random_sampling=False,
            classes=['benign', 'gleason grade 3', 'gleason grade 4', 'gleason grade 5']
        )
    )
)

optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='poly', power=0.9, min_lr=5e-4, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=200, metric='mDice')
