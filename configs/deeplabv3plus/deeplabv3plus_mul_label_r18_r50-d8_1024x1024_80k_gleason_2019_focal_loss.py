_base_ = 'deeplabv3plus_r50-d8_1024x1024_80k_gleason_2019_focal_loss.py'
model = dict(
    decode_head=dict(
        type='DepthwiseSeparableASPPMulLabelHead',
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
        loss_decode=dict(
            type='FocalLoss'
        ),
        loss_single=dict(
            type='FocalLoss'
        ),
        sigma=1,
        loss_step=1000))

data = dict(
    train=dict(
        ann_dir=['train/train/annotations', 'train/train/Maps1_T', 'train/train/Maps3_T', 'train/train/Maps4_T']))
