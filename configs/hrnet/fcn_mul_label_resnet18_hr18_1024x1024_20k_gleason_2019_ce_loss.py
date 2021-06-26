_base_ = './fcn_hr18_1024x1024_20k_gleason_2019_ce_loss.py'
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
            out_channels=4,
            kernel_size=3,
            padding=1),
        mul_label_ind=[1, 2, 3, 4],
        num_experts=4,
        final_label_ind=0,
        pretrained='torchvision://resnet18',
        num_classes=4,
        norm_cfg=norm_cfg,
        loss_decode=dict(type='CrossEntropyLoss'),
        loss_single=dict(type='CrossEntropyLoss'),
        sigma=1,
        fc_in_channels=512,
        loss_step=1000))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(ann_dir=[
        'train/annotations', 'train/Maps1_T', 'train/Maps3_T', 'train/Maps4_T',
        'train/he'
    ]))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
