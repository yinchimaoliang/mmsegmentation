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
            depth=18,
            num_stages=4,
            out_indices=(3,),
            style='pytorch'),
        wei_net_neck=dict(type='GlobalAveragePooling'),
        wei_net_in_channels=512,
        wei_net_out_channels=3,
        mul_label_ind=[1, 2, 3],
        final_label_ind=0,
        num_classes=4,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='DiceLoss'
        )
    )
)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        ann_dir=['train/3-labels/annotations', 'train/3-labels/Maps3_T', 'train/3-labels/Maps4_T', 'train/3-labels/Maps5_T']))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU')
