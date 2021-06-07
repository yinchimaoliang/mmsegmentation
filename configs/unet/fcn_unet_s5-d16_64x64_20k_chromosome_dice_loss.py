_base_ = './fcn_unet_s5-d16_64x64_20k_chromosome_ce_loss.py'
model = dict(decode_head=dict(loss_decode=dict(type='DiceLoss')))
