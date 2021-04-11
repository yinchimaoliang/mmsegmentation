import argparse
import os

import cv2 as cv
import mmcv
import numpy as np

from mmseg.apis import inference_segmentor, init_segmentor


def parse_args():
    parser = argparse.ArgumentParser(description='Inference a segmentor')
    parser.add_argument('config', help='inference config file path')
    parser.add_argument('--ckpt-dir', help='the dir of ckpt')
    parser.add_argument('--img-dir', help='the dir of images')
    parser.add_argument('--result-dir', help='the dir to save results')
    args = parser.parse_args()

    return args


def inference(config, ckpt_dir, img_dir, result_dir):

    mmcv.mkdir_or_exist(result_dir)
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config, ckpt_dir, device='cuda:0')

    names = os.listdir(img_dir)
    for name in names:
        result = inference_segmentor(model, os.path.join(img_dir, name))
        img = mmcv.imread(os.path.join(img_dir, name))
        mask = np.expand_dims(result[0], 2).repeat(3, 2)
        result = img * mask
        cv.imwrite(os.path.join(result_dir, name), result.astype(np.uint8))
        print(name)


if __name__ == '__main__':
    args = parse_args()
    config = args.config
    ckpt_dir = args.ckpt_dir
    img_dir = args.img_dir
    result_dir = args.result_dir
    inference(config, ckpt_dir, img_dir, result_dir)
