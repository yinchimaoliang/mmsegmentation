_base_ = './fcn_hr18_1024x1024_20k_gleason_2019_focal_loss.py'

model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='FocalLoss',
            gamma=0.1,
            gauss_scale=50,
            gauss_kernel=5,
            gauss_sigma=3)))
