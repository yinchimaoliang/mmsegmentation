_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/gleason_2019.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(
        norm_cfg=norm_cfg),
    decode_head=dict(
        norm_cfg=norm_cfg,
        num_classes=4),
    auxiliary_head=dict(
        norm_cfg=norm_cfg,
        num_classes = 4),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )
