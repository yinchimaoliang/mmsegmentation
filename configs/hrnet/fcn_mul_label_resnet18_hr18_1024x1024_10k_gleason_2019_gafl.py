_base_ = './fcn_mul_label_resnet18_hr18_1024x1024_10k_gleason_2019_ce_loss.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='GaussAttentionFocalLoss',
            use_sigmoid=False,
            gamma=0.1,
            gauss_scale=50,
            gauss_kernel=5,
            gauss_sigma=3),
        loss_single=dict(
            type='GaussAttentionFocalLoss',
            use_sigmoid=False,
            gamma=0.1,
            gauss_scale=50,
            gauss_kernel=5,
            gauss_sigma=3)))
