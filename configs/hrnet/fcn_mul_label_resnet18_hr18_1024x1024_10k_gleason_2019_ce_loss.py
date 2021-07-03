_base_ = './fcn_mul_label_resnet18_hr18_1024x1024_20k_gleason_2019_ce_loss.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(decode_head=dict(loss_step=1000))

data = dict(
    train=dict(ann_dir=[
        'train/annotations', 'train/Maps1_T', 'train/Maps3_T', 'train/Maps4_T',
        'train/he_high'
    ]))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
