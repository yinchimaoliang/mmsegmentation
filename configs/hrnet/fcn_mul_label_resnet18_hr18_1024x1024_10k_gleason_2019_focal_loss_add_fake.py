_base_ = './fcn_mul_label_resnet18_' \
    'hr18_1024x1024_10k_gleason_2019_ce_loss_add_fake.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    decode_head=dict(
        loss_decode=dict(type='FocalLoss', use_sigmoid=False, gamma=0.1),
        loss_single=dict(type='FocalLoss', use_sigmoid=False, gamma=0.1)))
