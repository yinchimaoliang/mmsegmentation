import argparse
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import numpy as np
import os
import cv2 as cv


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
        cv.imwrite(os.path.join(result_dir, name.replace('jpg', 'png')), result[0].astype(np.uint8))
        print(name.replace('jpg', 'png'))

    # test a single image and show the results
    # img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
    # result = inference_segmentor(model, img)


if __name__ == '__main__':
    args = parse_args()
    config = args.config
    ckpt_dir = args.ckpt_dir
    img_dir = args.img_dir
    result_dir = args.result_dir
    inference(config, ckpt_dir, img_dir, result_dir)
