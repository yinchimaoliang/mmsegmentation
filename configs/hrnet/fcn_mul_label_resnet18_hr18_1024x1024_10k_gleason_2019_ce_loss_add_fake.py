_base_ = './fcn_mul_label_resnet18_' \
    'hr18_1024x1024_20k_gleason_2019_ce_loss_add_fake.py'

runner = dict(type='IterBasedRunner', max_iters=10000)
model = dict(decode_head=dict(loss_step=1000))
