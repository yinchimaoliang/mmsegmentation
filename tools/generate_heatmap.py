import argparse
import os
import mmcv
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
from mmseg.datasets.pipelines import Normalize
from mmseg.models.losses.utils import gkern

mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]
to_rgb=True

def parse_args():
    parser = argparse.ArgumentParser(description='Inference a segmentor')
    parser.add_argument('--img-dir', help='the dir of images')
    parser.add_argument('--result-dir', help='the dir to save results')
    args = parser.parse_args()

    return args


def _process_img(img):
    results = dict(img = img)
    normalize = Normalize(mean=mean, std=std, to_rgb=to_rgb)
    results = normalize(results)
    img_normalized = results['img']
    return img_normalized

def _generate_heatmap(img, gauss_kernel, gauss_sigma, gauss_scale):
    kernel = gkern(gauss_kernel, gauss_sigma)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    img = img.mean(dim=1, keepdim=True)
    kernel = torch.from_numpy(kernel).to(img).expand(1, 1, gauss_kernel, gauss_kernel)
    img_blurred = F.conv2d(img, nn.Parameter(kernel, requires_grad=False), padding=(gauss_kernel - 1) // 2)
    img_weight = 1 + gauss_scale * torch.abs(img_blurred - torch.mean(img, dim=1, keepdim=True))
    img_weight = img_weight.squeeze(0)
    img_weight = img_weight.squeeze(0)
    img_weight = img_weight.detach().numpy()
    heatmap = cv.applyColorMap(np.uint8(img_weight * 255), cv.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    return heatmap

def _show_heatmap_on_img(img, heatmap, save_path):
    img = img / 255
    cam = np.float32(img) + heatmap
    cam = cam / np.max(cam)
    cv.imwrite(save_path, np.uint8(255*cam))


if __name__ == '__main__':
    args = parse_args()
    img_dir = args.img_dir
    result_dir = args.result_dir
    mmcv.mkdir_or_exist(result_dir)
    names = os.listdir(img_dir)
    for name in names:
        img = cv.imread(os.path.join(img_dir, name))
        img_normalized = _process_img(img)
        heatmap = _generate_heatmap(img_normalized, 5, 3, 5)
        cv.imwrite(os.path.join(result_dir, name), np.uint8(heatmap * 255))
        _show_heatmap_on_img(img, heatmap, os.path.join(result_dir, name))
        break