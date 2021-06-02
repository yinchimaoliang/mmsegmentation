_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/drive.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(num_classes=2, norm_cfg=norm_cfg),
    test_cfg=dict(crop_size=(64, 64), stride=(42, 42)))
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
