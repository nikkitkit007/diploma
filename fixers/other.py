import cv2
from typing import List
import os
from configurations import broken_dataset_path, origin_dataset_path
from utils.img_worker import find_diff_px

import numpy as np


class Other:

    @staticmethod
    def gauss_blur(img, x: int, y: int):        # todo read about this method

        filtered_img = cv2.GaussianBlur(img, (3, 3), 2.0)
        return filtered_img[x][y]

    @staticmethod
    def calc_dist(bgr1, bgr2):
        dist = ((int(bgr1[0]) - int(bgr2[0])) ** 2 + (int(bgr1[1]) - int(bgr2[1])) ** 2 + (
                int(bgr1[2]) - int(bgr2[2])) ** 2) ** 0.5
        return dist


def main():
    img_name_list = os.listdir(broken_dataset_path)[:2]

    for img_name in img_name_list:
        broken_img = cv2.imread(broken_dataset_path + img_name)
        origin_img = cv2.imread(origin_dataset_path + img_name)

        x, y = find_diff_px(img_name)
        bgr_origin = origin_img[x][y]  # BGR (b, g, r)
        bgr_broken = broken_img[x][y]  # BGR (b, g, r)

        print(f'origin: {bgr_origin}')
        res = Other.gauss_blur(origin_img, x, y)
        # res = Inpainter.navier_stokes(origin_img, x, y)
        print(res, '\n')


if __name__ == "__main__":
    main()
