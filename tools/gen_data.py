import argparse
import os
import os.path as osp
import random

import mmcv
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Generate data')
    parser.add_argument(
        '--source-path',
        default='./data/DRIVE/training',
        help='path of the source data')
    parser.add_argument(
        '--target-path',
        default='./data/DRIVE',
        help='path of the source data')
    parser.add_argument(
        '--train-ratio', type=float, default=0.7, help='ratio for training')
    args = parser.parse_args()

    return args


def spilt_data(source_path, ratio, target_path):
    img_names = os.listdir(osp.join(source_path, 'images'))
    random.shuffle(img_names)
    train_names = img_names[:int(ratio * len(img_names))]
    valid_names = img_names[int(ratio * len(img_names)):]
    train_img_path = osp.join(target_path, 'train', 'images')
    train_ann_path = osp.join(target_path, 'train', 'annotations')
    valid_img_path = osp.join(target_path, 'valid', 'images')
    valid_ann_path = osp.join(target_path, 'valid', 'annotations')
    mmcv.mkdir_or_exist(train_img_path)
    mmcv.mkdir_or_exist(train_ann_path)
    mmcv.mkdir_or_exist(valid_img_path)
    mmcv.mkdir_or_exist(valid_ann_path)
    for train_name in train_names:
        img = mmcv.imread(osp.join(source_path, 'images', train_name))
        ann = np.array(
            Image.open(
                osp.join(source_path, '1st_manual',
                         train_name.replace('training.tif', 'manual1.gif'))))
        mask = np.array(
            Image.open(
                osp.join(source_path, 'mask',
                         train_name.replace('.tif', '_mask.gif'))))
        ann[ann > 0] = 1
        mask[mask > 0] = 1
        ann *= mask
        mask = np.expand_dims(mask, 2).repeat(3, 2)
        img *= mask
        mmcv.imwrite(
            img,
            osp.join(target_path, 'train', 'images',
                     train_name.replace('training.tif', '.png')))
        mmcv.imwrite(
            ann,
            osp.join(target_path, 'train', 'annotations',
                     train_name.replace('training.tif', '.png')))
        print(f'{train_name} finished')

    for valid_name in valid_names:
        img = mmcv.imread(osp.join(source_path, 'images', valid_name))
        ann = np.array(
            Image.open(
                osp.join(source_path, '1st_manual',
                         valid_name.replace('training.tif', 'manual1.gif'))))
        mask = np.array(
            Image.open(
                osp.join(source_path, 'mask',
                         train_name.replace('.tif', '_mask.gif'))))
        ann[ann > 0] = 1
        mask[mask > 0] = 1
        ann *= mask
        mask = np.expand_dims(mask, 2).repeat(3, 2)
        img *= mask
        mmcv.imwrite(
            img,
            osp.join(target_path, 'valid', 'images',
                     valid_name.replace('training.tif', '.png')))
        mmcv.imwrite(
            ann,
            osp.join(target_path, 'valid', 'annotations',
                     valid_name.replace('training.tif', '.png')))
        print(f'{valid_name} finished')


if __name__ == '__main__':
    args = parse_args()
    source_path = args.source_path
    target_path = args.target_path
    train_ratio = args.train_ratio
    spilt_data(source_path, train_ratio, target_path)
