_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/chromosome.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg),
    auxiliary_head=dict(norm_cfg=norm_cfg),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
checkpoint_config = dict(by_epoch=False, interval=-1)
evaluation = dict(
    interval=2000, metric='mDice', save_best='mDice', rule='greater')
data = dict(samples_per_gpu=10)
