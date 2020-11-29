_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/pathology_full.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(num_classes=5, norm_cfg=norm_cfg),
    auxiliary_head=dict(
        num_classes=5,
        norm_cfg=norm_cfg,
        loss_decode=dict(type='DiceLoss', loss_weight=0.4)))

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)

evaluation = dict(interval=500, metric='mDice')
