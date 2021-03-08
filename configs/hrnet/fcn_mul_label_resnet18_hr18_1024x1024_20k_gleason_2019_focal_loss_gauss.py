_base_ = './fcn_mul_label_resnet18_hr18_1024x1024_' \
         '20k_gleason_2019_focal_loss.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    decode_head=dict(
        type='FCNMulLabelHead',
        in_channels=[18, 36, 72, 144],
        channels=sum([18, 36, 72, 144]),
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
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='FocalLoss',
            gamma=0.5,
            gauss_scale=50,
            gauss_kernel=5,
            gauss_sigma=3),
        loss_single=dict(
            type='FocalLoss',
            gamma=0.5,
            gauss_scale=50,
            gauss_kernel=5,
            gauss_sigma=3),
        sigma=1,
        loss_step=1000))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(ann_dir=[
        'train/train/annotations', 'train/train/Maps1_T',
        'train/train/Maps3_T', 'train/train/Maps4_T'
    ]))
