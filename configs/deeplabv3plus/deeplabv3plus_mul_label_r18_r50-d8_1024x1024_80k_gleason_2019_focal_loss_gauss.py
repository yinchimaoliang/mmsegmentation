_base_ = 'deeplabv3plus_mul_label_r18_r50-d8_1024x1024_80k_gleason_2019_focal_loss.py'
model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='FocalLoss', gauss_scale=50, gauss_kernel=5, gauss_sigma=3
        ),
        loss_single=dict(
            type='FocalLoss', gauss_scale=50, gauss_kernel=5, gauss_sigma=3
        )))

data = dict(
    train=dict(
        ann_dir=['train/train/annotations', 'train/train/Maps1_T', 'train/train/Maps3_T', 'train/train/Maps4_T']))
