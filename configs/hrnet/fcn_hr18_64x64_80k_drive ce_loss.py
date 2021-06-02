_base_ = './fcn_hr18_64x64_80k_drive ce_loss.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(num_classes=2, norm_cfg=norm_cfg),
    test_cfg=dict(crop_size=(64, 64), stride=(42, 42)))
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
checkpoint_config = dict(interval=-1, save_last=True)
evaluation = dict(metric='mDice', save_best='mDice', rule='greater')
