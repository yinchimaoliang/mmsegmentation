_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/gleason_2019.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg, ),
    decode_head=dict(
        type='FCNHead',
        num_classes=4,
        norm_cfg=norm_cfg,
        loss_decode=dict(type='FocalLoss', use_sigmoid=False, gamma=0.1)))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(ann_dir='train/train/annotations'))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='poly', power=0.1, min_lr=1e-6, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mDice')
