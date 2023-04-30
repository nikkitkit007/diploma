import os
from cv2 import cv2

from utils.heatmap import *
from utils.img_worker import find_diff_px

from configurations import Datasets
methods = {"mehalanobis": {"score": 0, "method": calculate_map_mehalanobis_distance},
           "outlier_detection": {"score": 0, "method": calculate_map_outlier_detection},
           "z_score": {"score": 0, "method": calculate_map_z_score},
           "histogram": {"score": 0, "method": calculate_map_histogram}}


# def heatmap(img):
#     methods = [calculate_map_mehalanobis_distance, calculate_map_outlier_detection, calculate_map_z_score,
#                calculate_map_histogram]
#     for m in methods:
#         heatmap = get_heatmap(img, m)
#         cv2.imwrite(str(m) + '_heatmap.png', heatmap)


def outliers(img, broken_px, percents):
    img_0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_1 = img[:, :, 0]
    img_2 = img[:, :, 1]
    img_3 = img[:, :, 2]

    for name, method in methods.items():
        potential_px0 = get_outliers(img_0, method['method'], percents)
        potential_px1 = get_outliers(img_1, method['method'], percents)
        potential_px2 = get_outliers(img_2, method['method'], percents)
        potential_px3 = get_outliers(img_3, method['method'], percents)
        potential_px = np.concatenate((potential_px0, potential_px1, potential_px2, potential_px3))
        if broken_px in potential_px:
            methods[name]['score'] += 1


img_name_list = os.listdir(Datasets.broken)
total_img = len(img_name_list)
print(f"total_img_count: {total_img}")
for i in range(80, 100):
    for img in img_name_list:
        broken_img = cv2.imread(Datasets.broken + img)
        broken_px = find_diff_px(img)
        outliers(broken_img, broken_px, i)
        print(f"i = {i}\n"
                  f"{[{'name': name, 'score': round(method['score']/total_img, 4)} for name, method in methods.items()]}")


