_base_ = './fcn_hr18_1024x1024_20k_gleason_2019_ce_loss.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
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
            out_indices=(3, ),
            style='pytorch'),
        wei_net_conv=dict(
            type='Conv2d',
            in_channels=512,
            out_channels=3,
            kernel_size=3,
            padding=1),
        mul_label_ind=[1, 2, 3],
        final_label_ind=0,
        pretrained='torchvision://resnet18',
        num_classes=4,
        fc_in_channels=32768,
        pool_kernel=32,
        norm_cfg=norm_cfg,
        loss_decode=dict(type='CrossEntropyLoss'),
        loss_single=dict(type='CrossEntropyLoss'),
        sigma=1,
        loss_step=1000))
