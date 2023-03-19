import cv2
import numpy as np
from typing import Tuple

origin_dataset_path = "/home/nikita/Desktop/one-pixel-attack-master/cifar10/test/"
broken_dataset_path = "/home/nikita/Desktop/one-pixel-attack-master/output/"


def compare(img1, img2):
    diff = cv2.subtract(img1, img2)
    return diff


def find_diff_px(img_name: str, show_imgs: bool = False) -> Tuple[int, int]:
    img_origin = cv2.imread(origin_dataset_path + img_name, flags=cv2.IMREAD_GRAYSCALE)
    img_broken = cv2.imread(broken_dataset_path + img_name, flags=cv2.IMREAD_GRAYSCALE)

    compared_pictures = compare(img_origin, img_broken)

    i, j = np.unravel_index(compared_pictures.argmax(), compared_pictures.shape)[:2]

    if show_imgs:
        cv2.imshow("origin", img_origin)
        cv2.imshow("broken", img_broken)
        cv2.imshow("dif", compared_pictures)
        cv2.waitKey(0)

    return i, j


if __name__ == '__main__':
    # img_name = str(input())
    img_name = "1335.png"
    # img_name = "11774.png"
    # img_name = "12195.png"

    print(find_diff_px(img_name))
