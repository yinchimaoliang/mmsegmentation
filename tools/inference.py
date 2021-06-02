import argparse
import os
import os.path as osp

import mmcv

from mmseg.apis import inference_segmentor, init_segmentor


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--img-dir',
        default='./data/DRIVE/test/images',
        help='dir of images to be inferenced')
    parser.add_argument('--target-dir', help='target dir to save the results.')
    args = parser.parse_args()
    return args


def inference_single(model, img_path, target_path):
    result = inference_segmentor(model, img_path)
    mmcv.imwrite(result[0], target_path)


def main():
    args = parse_args()
    cfg = args.config
    ckpt = args.checkpoint
    img_dir = args.img_dir
    target_dir = args.target_dir
    mmcv.mkdir_or_exist(target_dir)
    img_names = os.listdir(img_dir)
    model = init_segmentor(cfg, ckpt, device='cuda:0')
    for img_name in img_names:
        inference_single(model, osp.join(img_dir, img_name),
                         osp.join(target_dir, img_name))
        print(f'{img_name} finished.')


if __name__ == '__main__':
    main()
