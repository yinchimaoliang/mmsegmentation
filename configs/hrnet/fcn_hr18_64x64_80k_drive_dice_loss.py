_base_ = './fcn_hr18_64x64_80k_drive_ce_loss.py'
model = dict(decode_head=dict(loss_decode=dict(type='DiceLoss')))
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
checkpoint_config = dict(interval=-1, save_last=True)
evaluation = dict(metric='mDice', save_best='mDice', rule='greater')
