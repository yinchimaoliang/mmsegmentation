import argparse
import math

import cv2 as cv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--img-path', help='source image path')
    parser.add_argument('--target-path', help='target image path')
    args = parser.parse_args()

    return args


def grab_cut(img_path, target_path):
    img = cv.imread(img_path)
    mask = np.zeros(img.shape[:2], np.uint8)
    new_img = np.zeros_like(img)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    for i in range(math.ceil(img.shape[0] / 1000)):
        for j in range(math.ceil(img.shape[1] / 1000)):
            up = i * 1000
            left = j * 1000
            down = min(up + 1000, img.shape[0])
            right = min(left + 1000, img.shape[1])
            rect = [left, up, 1000, 1000]
            mask = np.zeros(img.shape[:2], np.uint8)
            cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5,
                       cv.GC_INIT_WITH_RECT)

            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            new_img[up:down,
                    left:right, :] = (img *
                                      mask2[:, :, np.newaxis])[up:down,
                                                               left:right, :]

            print(f'left:{left}, up:{up} finished')
    cv.imwrite(target_path, new_img)


if __name__ == '__main__':
    args = parse_args()
    grab_cut(args.img_path, args.target_path)
