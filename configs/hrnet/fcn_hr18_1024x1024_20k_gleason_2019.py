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
            type='SoftCrossEntropyLoss', use_mask=True, reduction='mean'),
        ignore_index=255
        ),

)