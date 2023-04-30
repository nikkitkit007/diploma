import numpy as np
import os
import cv2
from configurations import Datasets
from utils.img_worker import find_diff_px
from scipy.interpolate import interp2d
from utils.img_worker import calc_dist

proc = {}


class Interpolator:

    @staticmethod
    def nearest_neighbor(img, x: int, y: int) -> list:
        restored_px = img[x - 1][y - 1]
        # img[x, y] = restored_px

        return restored_px

    @staticmethod
    def bilinear(img, x: int, y: int) -> list:
        top_left = img[x - 1][y - 1]
        top_right = img[x - 1][y + 1]
        bottom_left = img[x + 1][y - 1]
        bottom_right = img[x + 1][y + 1]

        restored_px_r = int((int(top_left[2]) + int(top_right[2]) + int(bottom_left[2]) + int(bottom_right[2])) / 4)
        restored_px_g = int((int(top_left[1]) + int(top_right[1]) + int(bottom_left[1]) + int(bottom_right[1])) / 4)
        restored_px_b = int((int(top_left[0]) + int(top_right[0]) + int(bottom_left[0]) + int(bottom_right[0])) / 4)
        restored_px = [restored_px_b, restored_px_g, restored_px_r]

        return restored_px

    @staticmethod
    def bicubic(img, x: int, y: int):
        """
        Perform bicubic interpolation to restore a missing pixel in an image.
        """

        def cubic_convolution_coefficient(xx):
            """
            Compute the cubic convolution coefficients for a given distance x.
            """
            abs_x = np.abs(xx)
            if abs_x <= 1:
                return 1 - 2 * abs_x ** 2 + abs_x ** 3
            elif abs_x < 2:
                return 4 - 8 * abs_x + 5 * abs_x ** 2 - abs_x ** 3
            else:
                return 0

        # Convert the image to a NumPy array
        original_array = np.copy(img)
        original_array[x][y] = Interpolator.bilinear(img, x, y)
        # Define the row and column indices of the neighboring pixels
        row_indices = [x - 1, x, x + 1, x + 2]
        col_indices = [y - 1, y, y + 1, y + 2]

        # Extract the pixel values of the neighboring pixels for each color channel
        Z_red = original_array[row_indices][:, col_indices, 2]
        Z_green = original_array[row_indices][:, col_indices, 1]
        Z_blue = original_array[row_indices][:, col_indices, 0]

        # Compute the weights for the neighboring pixels using cubic convolution
        weights_red = np.zeros((4, 4))
        weights_green = np.zeros((4, 4))
        weights_blue = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                x_dist = (col_indices[j] - y) / 3
                y_dist = (row_indices[i] - x) / 3
                weights_red[i, j] = cubic_convolution_coefficient(x_dist) * cubic_convolution_coefficient(y_dist)
                weights_green[i, j] = cubic_convolution_coefficient(x_dist) * cubic_convolution_coefficient(y_dist)
                weights_blue[i, j] = cubic_convolution_coefficient(x_dist) * cubic_convolution_coefficient(y_dist)

        # Reshape the weights and pixel values to facilitate matrix multiplication
        weights_red = weights_red.reshape((-1, 1))
        weights_green = weights_green.reshape((-1, 1))
        weights_blue = weights_blue.reshape((-1, 1))
        Z_red = Z_red.flatten()
        Z_green = Z_green.flatten()
        Z_blue = Z_blue.flatten()

        Z2_red = np.dot(weights_red.T, Z_red)
        Z2_green = np.dot(weights_green.T, Z_green)
        Z2_blue = np.dot(weights_blue.T, Z_blue)

        restored_px = [int(Z2_blue) / 16, int(Z2_green) / 16, int(Z2_red) / 16]

        return restored_px

    @staticmethod
    def lanczos(img, x: int, y: int, window_size=3, a=2):
        def lanczos_kernel(xx):
            if xx == 0:
                return 1
            elif abs(xx) < window_size:
                # return aa * np.sin(np.pi * xx) * np.sin(np.pi * xx / aa) / (np.pi ** 2 * xx ** 2)
                return a * np.sinc(xx) * np.sinc(xx / a)
            else:
                return 0

        original_array = np.copy(img)
        original_array[x][y] = Interpolator.bilinear(original_array, x, y)
        # Compute the range of pixels to consider for interpolation
        xmin = int(x) - window_size
        xmax = int(x) + window_size + 1
        ymin = int(y) - window_size
        ymax = int(y) + window_size + 1

        # Compute the numerator and denominator of the interpolation equation
        numerator = 0
        denominator = 0
        rgb = [2, 1, 0]
        interpolated_pixel = [0, 0, 0]
        for chanel in rgb:
            for i in range(xmin, xmax):
                for j in range(ymin, ymax):
                    # Compute the weight for the current pixel
                    wx = lanczos_kernel(x - i)
                    wy = lanczos_kernel(y - j)
                    w = wx * wy
                    # Add the weighted pixel value to the numerator and denominator
                    if 0 <= i < original_array.shape[0] and 0 <= j < original_array.shape[1]:
                        numerator += w * original_array[i][j][chanel]
                        denominator += w

            # Compute the interpolated pixel value
            interpolated_pixel[chanel] = numerator / denominator

        return [round(res, 3) for res in interpolated_pixel]

    @staticmethod
    def custom(img, x: int, y: int):

        row_indices = [x - 1, x, x + 1]
        col_indices = [y - 1, y, y + 1]

        Z_bgr = img[row_indices][:, col_indices]

        line_px = []
        for i in range(3):
            for j in range(3):
                if not (i == 1 and j == 1):
                    line_px.append(Z_bgr[i][j])

        distances = []

        distances.append([calc_dist(Z_bgr[0][0], Z_bgr[2][2]),
                          [(int(a) + int(b)) / 2 for a, b in zip(Z_bgr[0][0], Z_bgr[2][2])]])
        distances.append([calc_dist(Z_bgr[0][1], Z_bgr[2][1]),
                          [(int(a) + int(b)) / 2 for a, b in zip(Z_bgr[0][0], Z_bgr[2][2])]])
        distances.append([calc_dist(Z_bgr[0][2], Z_bgr[2][0]),
                          [(int(a) + int(b)) / 2 for a, b in zip(Z_bgr[0][0], Z_bgr[2][2])]])
        distances.append([calc_dist(Z_bgr[1][0], Z_bgr[1][2]),
                          [(int(a) + int(b)) / 2 for a, b in zip(Z_bgr[0][0], Z_bgr[2][2])]])

        min_dist = sorted(distances, key=lambda x: x[0])[0]
        return min_dist[1]


interpolator = {"nearest_neighbor": {'method': Interpolator.nearest_neighbor, "score": 0},
                "bilinear": {'method': Interpolator.bilinear, "score": 0},
                "bicubic": {'method': Interpolator.bicubic, "score": 0},
                "lanczos": {'method': Interpolator.lanczos, "score": 0},
                "custom": {'method': Interpolator.custom, "score": 0}}


def main():
    for name, method in interpolator.items():
        proc[name] = 0

    img_name_list = os.listdir(Datasets.broken)[:2]

    for img_name in img_name_list:
        broken_img = cv2.imread(Datasets.broken + img_name)
        origin_img = cv2.imread(Datasets.origin + img_name)

        x, y = find_diff_px(img_name)
        bgr_origin = origin_img[x][y]  # BGR (b, g, r)

        print(f'origin: {bgr_origin}')
        res = Interpolator.bilinear(origin_img, x, y)
        print(res, '\n')


if __name__ == '__main__':
    main()
