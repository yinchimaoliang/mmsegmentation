import argparse
import os
import os.path as osp
import random

import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--data-path', help='train config file path')
    parser.add_argument(
        '--target-path', help='the dir to save logs and models')
    parser.add_argument(
        '--train-ratio', type=float, help='the dir to save logs and models')
    args = parser.parse_args()

    return args


def main(data_path, target_path, train_ratio):
    names = os.listdir(data_path)
    random.shuffle(names)
    train_names = names[:int(len(names) * train_ratio)]
    valid_names = names[int(len(names) * train_ratio):]
    mmcv.mkdir_or_exist(osp.join(target_path, 'train', 'images'))
    mmcv.mkdir_or_exist(osp.join(target_path, 'train', 'annotations'))
    mmcv.mkdir_or_exist(osp.join(target_path, 'valid', 'images'))
    mmcv.mkdir_or_exist(osp.join(target_path, 'valid', 'annotations'))
    for train_name in train_names:
        train_img = mmcv.imread(
            osp.join(data_path, train_name, 'images', f'{train_name}.png'))
        ann_names = os.listdir(osp.join(data_path, train_name, 'masks'))
        for i, ann_name in enumerate(ann_names):
            ann = mmcv.imread(
                osp.join(data_path, train_name, 'masks', ann_name), 0)
            print(ann_name)
            poses = np.where(ann > 0)
            up = poses[0].min()
            down = poses[0].max()
            left = poses[1].min()
            right = poses[1].max()
            sub_img = train_img[up:down + 1, left:right + 1, :]
            sub_ann = ann[up:down + 1, left:right + 1]
            sub_ann[sub_ann > 0] = 1
            mmcv.imwrite(
                sub_img,
                osp.join(target_path, 'train', 'images',
                         train_name.split('.')[0] + '_' + str(i) + '.png'))
            mmcv.imwrite(
                sub_ann,
                osp.join(target_path, 'train', 'annotations',
                         train_name.split('.')[0] + '_' + str(i) + '.png'))

    for valid_name in valid_names:
        valid_img = mmcv.imread(
            osp.join(data_path, valid_name, 'images', f'{valid_name}.png'))
        ann_names = os.listdir(osp.join(data_path, valid_name, 'masks'))
        for i, ann_name in enumerate(ann_names):
            ann = mmcv.imread(
                osp.join(data_path, valid_name, 'masks', ann_name), 0)
            print(ann_name)
            poses = np.where(ann > 0)
            up = poses[0].min()
            down = poses[0].max()
            left = poses[1].min()
            right = poses[1].max()
            sub_img = valid_img[up:down + 1, left:right + 1, :]
            sub_ann = ann[up:down + 1, left:right + 1]
            sub_ann[sub_ann > 0] = 1
            mmcv.imwrite(
                sub_img,
                osp.join(target_path, 'train', 'images',
                         valid_name.split('.')[0] + '_' + str(i) + '.png'))
            mmcv.imwrite(
                sub_ann,
                osp.join(target_path, 'train', 'annotations',
                         valid_name.split('.')[0] + '_' + str(i) + '.png'))


if __name__ == '__main__':
    args = parse_args()
    main(args.data_path, args.target_path, args.train_ratio)
