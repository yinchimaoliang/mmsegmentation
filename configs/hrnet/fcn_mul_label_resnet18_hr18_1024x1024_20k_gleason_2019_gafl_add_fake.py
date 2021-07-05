_base_ = './fcn_mul_label_resnet18_' \
    'hr18_1024x1024_20k_gleason_2019_gafl.py'

data = dict(
    train=dict(
        img_dir='train/add_fake/images',
        ann_dir=[
            'train/add_fake/annotations', 'train/add_fake/Maps1_T',
            'train/add_fake/Maps3_T', 'train/add_fake/Maps4_T',
            'train/add_fake/he_high'
        ]))
