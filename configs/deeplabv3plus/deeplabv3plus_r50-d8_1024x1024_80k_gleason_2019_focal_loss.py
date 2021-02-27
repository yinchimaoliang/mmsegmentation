_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/gleason_2019.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(
        norm_cfg=norm_cfg),
    decode_head=dict(
        align_corners=True,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='FocalLoss'
        ),
        num_classes=4,
    ),

    auxiliary_head=None,
    test_cfg=dict(mode='whole'))

lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mDice')