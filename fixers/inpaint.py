import cv2
import os
from configurations import broken_dataset_path, origin_dataset_path
from utils.img_worker import find_diff_px

import numpy as np


class Inpainter:

    @staticmethod
    def navier_stokes(img, x: int, y: int):

        mask = np.zeros((32, 32), np.uint8)
        mask[x, y] = 1
        inpaint = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

        return inpaint[x][y]

    @staticmethod
    def telea(img, x: int, y: int):

        mask = np.zeros((32, 32), np.uint8)
        mask[x, y] = 1
        inpaint = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        return inpaint[x][y]


def main():
    img_name_list = os.listdir(broken_dataset_path)[:2]

    for img_name in img_name_list:
        broken_img = cv2.imread(broken_dataset_path + img_name)
        origin_img = cv2.imread(origin_dataset_path + img_name)

        x, y = find_diff_px(img_name)
        bgr_origin = origin_img[x][y]  # BGR (b, g, r)
        bgr_broken = broken_img[x][y]  # BGR (b, g, r)

        print(f'origin: {bgr_origin}')
        res = Inpainter.telea(origin_img, x, y)
        # res = Inpainter.navier_stokes(origin_img, x, y)
        print(res, '\n')


if __name__ == "__main__":
    main()
