_base_ = 'deeplabv3plus_r50-d8_1024x1024_80k_gleason_2019_focal_loss.py'
model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='FocalLoss'
        ),
        loss_single=dict(
            type='FocalLoss'
        )))

data = dict(
    train=dict(
        ann_dir=['train/train/annotations', 'train/train/Maps1_T', 'train/train/Maps3_T', 'train/train/Maps4_T']))
