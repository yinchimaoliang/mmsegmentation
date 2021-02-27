_base_ = 'deeplabv3plus_r50-d8_1024x1024_80k_gleason_2019_focal_loss.py'
model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='FocalLoss', gauss_scale=50, gauss_kernel=5, gauss_sigma=3
        )))