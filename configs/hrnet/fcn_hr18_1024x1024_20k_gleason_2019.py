_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/gleason_2019.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(
        norm_cfg=norm_cfg,
        ),
    decode_head=dict(
        num_classes=4,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                 class_weight=[0.1, 1, 1, 5], gauss_scale=1, gauss_kernel=9, gauss_sigma=9
        )
    )
)