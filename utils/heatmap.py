import cv2
from outlier_detector import *
import numpy as np


def my_line_interpolation(mat):
    min_val = np.min(mat)
    max_val = np.max(mat)
    mat = 255 * (mat - min_val) / (max_val - min_val)
    return mat


def numpy_line_interpolation(mat):
    min_val = np.min(mat)
    max_val = np.max(mat)
    dist_min = 0
    dist_max = 255
    return np.interp(mat, (min_val, max_val), (dist_min, dist_max))


def histogram_interpolation(mat):  # Делает по Нр но  фигня получается, лучше использовать другие методы
    src_min, src_max = np.min(mat), np.max(mat)
    # Определение количества бинов для гистограммы
    num_bins = 256
    # Вычисление гистограммы массива
    hist, bin_edges = np.histogram(mat, bins=num_bins, range=(src_min, src_max))
    # Нормализация гистограммы до диапазона 0-255
    hist_norm = (hist - np.min(hist)) / (np.max(hist) - np.min(hist)) * 255

    # Преобразование значений массива путем замены каждого элемента на значение соответствующего бина гистограммы
    new_arr = np.zeros_like(mat)
    for i in range(num_bins - 1):
        mask = np.logical_and(mat >= bin_edges[i], mat < bin_edges[i + 1])
        new_arr[mask] = hist_norm[i]
    new_arr = new_arr.astype(np.uint8)
    return new_arr


def get_outliers(matrix, method, percentile):
    '''
    Врзвращает два кортежа
    '''
    m1 = numpy_line_interpolation(method(matrix))
    grey = np.round(m1).astype(np.uint8)

    threshold = np.percentile(grey, percentile)

    # Поиск ярких выбросов
    indices = np.where(grey > threshold)
    if indices:
        outliers = np.column_stack((indices[0], indices[1]))
        return outliers
    else:
        return []


def get_heatmap(matrix, method):
    """
    matrix is cv2 imread image, with cv2.IMREAD_GRAYSCALE
    `matrix = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)`
    method is any of outlier_detector calculate map methods:
        calculate_map_mehalanobis_distance;
        calculate_map_outlier_detection;
        calculate_map_z_score;
        calculate_map_histogram;
    """
    m1 = numpy_line_interpolation(method(matrix))
    grey = np.round(m1).astype(np.uint8)
    heatmap = cv2.applyColorMap(grey, cv2.COLORMAP_JET)
    return heatmap
