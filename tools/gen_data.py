import argparse
import os
import os.path as osp

import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Generate annotations data')
    parser.add_argument('--origin-path', help='original data path')
    parser.add_argument('--target-path', help='target data path')
    parser.add_argument(
        '--vote-folders',
        type=list,
        default=[
            'Maps1_T', 'Maps3_T', 'Maps4_T', 'he_high', 'Maps2_T', 'Maps5_T',
            'Maps6_T'
        ],
        help='annotations folders involved in voting')
    parser.add_argument('--num-classes', default=4, help='number of classes')
    parser.add_argument(
        '--rule', type=str, default='high', help='number of classes')

    args = parser.parse_args()

    return args


def gen_vote(origin_path, vote_folders, target_path, num_classes, rule):
    img_names = os.listdir(os.path.join(origin_path, 'images'))
    for img_name in img_names:
        anns = []
        img = mmcv.imread(osp.join(origin_path, 'images', img_name))
        for vote_folder in vote_folders:
            assert os.path.exists(
                os.path.join(origin_path, vote_folder, img_name))
            ann = mmcv.imread(osp.join(origin_path, vote_folder, img_name), 0)
            ann = mmcv.imresize_like(ann, img)
            anns.append(ann)

        ann = np.stack(anns)
        ann_count = np.zeros((num_classes, img.shape[0], img.shape[1]))
        for i in range(num_classes):
            ann_count[i, ...] = (ann == i).sum(axis=0)
        if rule == 'low':
            ann = ann_count.argmax(0)
        else:
            ann = ann_count[::-1, ...].argmax(0)
            ann = num_classes - 1 - ann

        mmcv.imwrite(ann, osp.join(target_path, img_name))
        print(f'{img_name} finished')


def resize_he(img_path, he_path):
    img_names = os.listdir(img_path)
    for img_name in img_names:
        he_img = mmcv.imread(osp.join(he_path, img_name), 0)
        img = mmcv.imread(osp.join(img_path, img_name))
        he_img = mmcv.imresize_like(he_img, img)
        mmcv.imwrite(he_img, osp.join(he_path, img_name))
        print(f'{img_name} finished.')


def gen_data(origin_path, target_path):
    names = os.listdir(origin_path)
    for name in names:
        ann = mmcv.imread(osp.join(origin_path, name), 0)
        ann[ann == 3] = 2
        mmcv.imwrite(ann, osp.join(target_path, name))
        print(f'{name} finished.')


def main():
    args = parse_args()
    origin_path = args.origin_path
    target_path = args.target_path
    vote_folders = args.vote_folders
    num_classes = args.num_classes
    rule = args.rule
    # mmcv.mkdir_or_exist(target_path)
    resize_he(
        osp.join(origin_path, 'images'), osp.join(origin_path, 'he_high'))
    gen_vote(origin_path, vote_folders, target_path, num_classes, rule)


if __name__ == '__main__':
    main()
