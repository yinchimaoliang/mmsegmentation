_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/drive.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg),
    auxiliary_head=dict(norm_cfg=norm_cfg, ),
    test_cfg=dict(crop_size=(64, 64), stride=(42, 42)))

checkpoint_config = dict(interval=-1, save_last=True)
evaluation = dict(metric='mDice', save_best='mDice', rule='greater')
