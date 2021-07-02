_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/gleason_2019.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg, ),
    decode_head=dict(type='FCNHead', num_classes=4, norm_cfg=norm_cfg))

data = dict(samples_per_gpu=2, workers_per_gpu=2)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
checkpoint_config = dict(by_epoch=False, save_last=True, interval=-1)
runner = dict(type='IterBasedRunner')
evaluation = dict(
    interval=2000, metric='mDice', save_best='mDice', rule='greater')
