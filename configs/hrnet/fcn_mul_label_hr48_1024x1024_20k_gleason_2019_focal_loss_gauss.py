_base_ = './fcn_hr18_1024x1024_20k_gleason_2019.py'
norm_cfg = dict(type='BN', requires_grad=True)
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
        sigma=1,
        loss_step=1000
    )
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_dir=['train/train/annotations', 'train/train/Maps1_T', 'train/train/Maps3_T', 'train/train/Maps4_T']))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=200, metric='mDice')
