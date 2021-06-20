_base_ = './fcn_hr18_1024x1024_20k_gleason_2019_3class_ce_loss.py'

model = dict(
    decode_head=dict(
        loss_decode=dict(type='FocalLoss', use_sigmoid=False, gamma=0.1)))
