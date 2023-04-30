import cv2
import os
from configurations import Datasets
from utils.img_worker import find_diff_px
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

import numpy as np


class Other:

    @staticmethod
    def gauss_blur(img, x: int, y: int):

        filtered_img = cv2.GaussianBlur(img, (3, 3), 2.0)
        return filtered_img[x][y]

    @staticmethod
    def fourier_transform(img, x: int, y: int):
        def restore_channel(channel, corrupted_coords):
            dft_channel = fft2(channel)
            dft_channel_shifted = fftshift(dft_channel)

            center = np.array(dft_channel_shifted.shape) // 2
            dft_corrupted_coords = (center[0] - corrupted_coords[0], center[1] - corrupted_coords[1])

            dft_channel_shifted[dft_corrupted_coords] = 0
            dft_channel = ifftshift(dft_channel_shifted)

            restored_channel = np.abs(ifft2(dft_channel))
            return restored_channel

        corrupted_b, corrupted_g, corrupted_r = cv2.split(img)

        restored_b = restore_channel(corrupted_b, (x, y))
        restored_g = restore_channel(corrupted_g, (x, y))
        restored_r = restore_channel(corrupted_r, (x, y))

        # Объединяем восстановленные цветовые каналы
        restored_image = cv2.merge((restored_b, restored_g, restored_r)).astype(np.uint8)
        return restored_image[x][y]


def main():
    img_name_list = os.listdir(Datasets.broken)[:2]

    for img_name in img_name_list:
        broken_img = cv2.imread(Datasets.broken + img_name)
        origin_img = cv2.imread(Datasets.origin + img_name)

        x, y = find_diff_px(img_name)
        bgr_origin = origin_img[x][y]  # BGR (b, g, r)

        print(f'origin: {bgr_origin}')
        res = Other.gauss_blur(broken_img, x, y)
        print(res, '\n')


if __name__ == "__main__":
    main()
