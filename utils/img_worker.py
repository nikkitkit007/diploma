import cv2
import os

import numpy as np
from typing import Tuple
import outlier_detector

from configurations import Datasets


def compare(img1, img2):
    diff = cv2.subtract(img1, img2)
    return diff


def calc_dist(bgr1, bgr2):
    dist = ((int(bgr1[0]) - int(bgr2[0])) ** 2 + (int(bgr1[1]) - int(bgr2[1])) ** 2 + (
            int(bgr1[2]) - int(bgr2[2])) ** 2) ** 0.5
    return dist


def find_diff_px(img_name: str, dataset_1: str = Datasets.origin, dataset_2: str = Datasets.broken,
                 show_imgs: bool = False) -> Tuple[int, int]:
    img_origin = cv2.imread(dataset_1 + img_name, flags=cv2.IMREAD_GRAYSCALE)
    img_broken = cv2.imread(dataset_2 + img_name, flags=cv2.IMREAD_GRAYSCALE)

    compared_pictures = compare(img_origin, img_broken)

    i, j = np.unravel_index(compared_pictures.argmax(), compared_pictures.shape)[:2]

    if show_imgs:
        cv2.imshow("origin", img_origin)
        cv2.imshow("broken", img_broken)
        cv2.imshow("dif", compared_pictures)
        cv2.waitKey(0)

    return i, j


def get_image_names(path: str, count=None):
    return os.listdir(path)[:count]


if __name__ == '__main__':
    img_name = "1335.png"
    # img_name = "11774.png"
    # img_name = "12195.png"
    im = cv2.imread(broken_dataset_path + img_name)
    cv2.imshow("T", im)
    cv2.waitKey(1)
    res = outlier_detector.z_score(broken_dataset_path + img_name, 3)
    print(find_diff_px(img_name))
