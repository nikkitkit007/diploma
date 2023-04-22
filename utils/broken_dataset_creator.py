from typing import Union
import os
import cv2
from random import randint

from configurations import origin_dataset_path, test_dataset_path


def create_test_dataset(src: str = origin_dataset_path, dst: str = test_dataset_path, count: Union[int, None] = None):
    img_names = os.listdir(src)[:count]

    for img_name in img_names:
        img = cv2.imread(src + img_name)
        img[randint(0, 31), randint(0, 31)] = (randint(0, 255), randint(0, 255), randint(0, 255))
        r = cv2.imwrite(dst + img_name, img)
        assert r is True
