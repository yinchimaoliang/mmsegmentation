_base_ = './fcn_hr18_1024x1024_20k_gleason_2019_ce_loss.py'

data = dict(
    train=dict(
        img_dir='train/add_fake/images', ann_dir='train/add_fake/annotations'))
