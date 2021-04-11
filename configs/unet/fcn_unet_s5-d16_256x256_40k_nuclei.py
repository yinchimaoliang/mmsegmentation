_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/nuclei.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=500, metric='mDice')

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg),
    auxiliary_head=dict(norm_cfg=norm_cfg),
    test_cfg=dict(mode='whole'))

data = dict(samples_per_gpu=10, workers_per_gpu=10)
