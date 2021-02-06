import argparse
import os
import mmcv
import shutil
import os.path as osp
import cv2 as cv
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='show gt')
    parser.add_argument('--label-dir', help='dir of labels', default='/home1/yinhaoli/data/gleason_2019/combined')
    parser.add_argument('--image-dir', help='dir of labels', default='/home1/yinhaoli/data/gleason_2019/train_imgs')
    parser.add_argument('--target-dir', help='dir of labels', default='/home1/yinhaoli/data/gleason_2019')
    parser.add_argument(
        '--train-ratio', default=0.7, type=int, help='ratio of the train number')
    args = parser.parse_args()

    return args

def _generate_division(img_dir, ann_dir, target_dir, train_ratio):
    mmcv.mkdir_or_exist(osp.join(target_dir, 'train', 'images'))
    mmcv.mkdir_or_exist(osp.join(target_dir, 'train', 'annotations'))
    mmcv.mkdir_or_exist(osp.join(target_dir, 'valid', 'images'))
    mmcv.mkdir_or_exist(osp.join(target_dir, 'valid', 'annotations'))
    mmcv.mkdir_or_exist(osp.join(target_dir, 'trainval', 'images'))
    mmcv.mkdir_or_exist(osp.join(target_dir, 'trainval', 'annotations'))
    names = os.listdir(img_dir)
    np.random.shuffle(names)
    train_names = names[:int(len(names) * train_ratio)]
    val_names = names[int(len(names) * train_ratio):]
    for train_name in train_names:
        print(train_name)
        img = mmcv.imread(osp.join(img_dir, train_name))
        mmcv.imwrite(img, osp.join(target_dir, 'train', 'images', train_name.split('.')[0]+'.png'))
        shutil.copyfile(osp.join(ann_dir, train_name.split('.')[0]+'.png'), osp.join(target_dir, 'train', 'annotations', train_name.split('.')[0]+'.png'))
    for val_name in val_names:
        print(val_name)
        img = mmcv.imread(osp.join(img_dir, val_name))
        mmcv.imwrite(img, osp.join(target_dir, 'valid', 'images', val_name.split('.')[0]+'.png'))
        shutil.copyfile(osp.join(ann_dir, val_name.split('.')[0]+'.png'), osp.join(target_dir, 'valid', 'annotations', val_name.split('.')[0]+'.png'))
    for name in names:
        print(name)
        img = mmcv.imread(osp.join(img_dir, name))
        mmcv.imwrite(img, osp.join(target_dir, 'trainval', 'images', name.split('.')[0]+'.png'))
        shutil.copyfile(osp.join(ann_dir, name.split('.')[0]+'.png'), osp.join(target_dir, 'trainval', 'annotations', name.split('.')[0]+'.png'))

def main():
    args = parse_args()
    label_dir = args.label_dir
    image_dir = args.image_dir
    target_dir = args.target_dir
    train_ratio = args.train_ratio
    _generate_division(image_dir, label_dir, target_dir, train_ratio)


if __name__ == '__main__':
    main()