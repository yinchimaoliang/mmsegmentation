import argparse
import os
import os.path as osp

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description='Generate annotations data')
    parser.add_argument('--origin-path', help='train config file path')
    parser.add_argument(
        '--target-path', help='the dir to save logs and models')

    args = parser.parse_args()

    return args


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
    mmcv.mkdir_or_exist(target_path)
    gen_data(origin_path, target_path)


if __name__ == '__main__':
    main()
