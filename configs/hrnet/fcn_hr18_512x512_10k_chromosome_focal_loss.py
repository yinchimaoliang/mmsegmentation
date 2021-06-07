_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/chromosome.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg, ),
    decode_head=dict(
        norm_cfg=norm_cfg, num_classes=2, loss_decode=dict(type='FocalLoss')))

runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=-1)
evaluation = dict(
    interval=1000, metric='mDice', save_best='mDice', rule='greater')
