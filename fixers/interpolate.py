import numpy as np
import os
import cv2
from configurations import broken_dataset_path, origin_dataset_path
from utils.img_worker import find_diff_px


class Interpolator:
    # todo make all combinations

    @staticmethod
    def compare_fix_methods(img_name):
        broken_img = cv2.imread(broken_dataset_path + img_name)
        origin_img = cv2.imread(origin_dataset_path + img_name)

        x, y = find_diff_px(img_name)
        bgr_origin = origin_img[x][y]  # BGR (b, g, r)
        bgr_broken = broken_img[x][y]  # BGR (b, g, r)

        bicubic = Interpolator.bicubic(broken_img, x, y)
        bilinear = Interpolator.bilinear(broken_img, x, y)
        nearest_neighbor = Interpolator.nearest_neighbor(broken_img, x, y)
        lanczos = Interpolator.lanczos(broken_img, x, y)
        custom = Interpolator.custom(broken_img, x, y)

        res = {"nearest_neighbor": nearest_neighbor,
               "bilinear": bilinear,
               "bicubic": bicubic,
               "lanczos": lanczos,
               "custom": custom}

        dist = {}
        for key in res:
            dist[key] = Interpolator.calc_dist(res[key], bgr_origin)
        dist = dict(sorted(dist.items(), key=lambda x: x[1]))
        print(f"\n\t<<<<{img_name}>>>>\n"
              f"original px: {bgr_origin}\n"
              f"nearest_neighbor: {nearest_neighbor}\n"
              f"bilinear: {bilinear}\n"
              f"bicubic: {bicubic}\n"
              f"lanczos: {lanczos}\n"
              f"custom: {custom}")
        #
        # print(f"\nTop of methods:\n{dist}")
        return dist

    @staticmethod
    def nearest_neighbor(img, x: int, y: int) -> list:
        restored_px = img[x-1][y-1]
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

        restored_px = [int(Z2_blue)/16, int(Z2_green)/16, int(Z2_red)/16]

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
    def calc_dist(bgr1, bgr2):
        dist = ((int(bgr1[0])-int(bgr2[0]))**2 + (int(bgr1[1])-int(bgr2[1]))**2 + (int(bgr1[2])-int(bgr2[2]))**2)**0.5
        return dist

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

        distances.append([Interpolator.calc_dist(Z_bgr[0][0], Z_bgr[2][2]), [(int(a)+int(b))/2 for a, b in zip(Z_bgr[0][0], Z_bgr[2][2])]])
        distances.append([Interpolator.calc_dist(Z_bgr[0][1], Z_bgr[2][1]), [(int(a)+int(b))/2 for a, b in zip(Z_bgr[0][0], Z_bgr[2][2])]])
        distances.append([Interpolator.calc_dist(Z_bgr[0][2], Z_bgr[2][0]), [(int(a)+int(b))/2 for a, b in zip(Z_bgr[0][0], Z_bgr[2][2])]])
        distances.append([Interpolator.calc_dist(Z_bgr[1][0], Z_bgr[1][2]), [(int(a)+int(b))/2 for a, b in zip(Z_bgr[0][0], Z_bgr[2][2])]])

        min_dist = sorted(distances, key=lambda x: x[0])[0]
        return min_dist[1]


def main():
    img_name_list = os.listdir(broken_dataset_path)[:2]
    results = {"nearest_neighbor": 0,
               "bilinear": 0,
               "bicubic": 0,
               "lanczos": 0,
               "custom": 0}
    for img_name in img_name_list:
        try:
            top_methods = Interpolator.compare_fix_methods(img_name)
            for rate, method in enumerate(top_methods):
                results[method] += rate
        except:
            pass

    print(f"\nTOP of methods: {dict(sorted(results.items(), key=lambda x: x[1]))}")


if __name__ == '__main__':
    main()