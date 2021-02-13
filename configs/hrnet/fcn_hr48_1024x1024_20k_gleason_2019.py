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
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384]),
        num_classes=4,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='DiceLoss', class_weight=[0.1, 1, 1, 10], gauss_scale=5, gauss_kernel=9, gauss_sigma=9
        )
))
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU')
