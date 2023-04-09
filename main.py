import cv2
from outlier_detector import *
import os

from utils.img_worker import find_diff_px
from utils.heatmap import get_heatmap, get_outliers
from configurations import broken_dataset_path, origin_dataset_path
import fixers.interpolate as interpolate

methods = {"mehalanobis": {"score": 0, "method": calculate_map_mehalanobis_distance},
           "outlier_detection": {"score": 0, "method": calculate_map_outlier_detection},
           "z_score": {"score": 0, "method": calculate_map_z_score},
           "histogram": {"score": 0, "method": calculate_map_histogram}}


def find_min_pixels_count(method, data, m1, m2, max_iterations=15):
    res_m1 = method(data, m1)
    res_m2 = method(data, m2)
    iter = 0
    res = res_m1
    mean = None
    while len(res) != 1 and iter < max_iterations:
        mean = (m1 + m2) / 2
        res_mean = method(data, mean)
        if len(res_mean) > 1:
            m1 = mean
        elif len(res_mean) == 0:
            m2 = mean
        res = res_m1
        iter += 1
    return mean


def heatmap(img):
    methods = [calculate_map_mehalanobis_distance, calculate_map_outlier_detection, calculate_map_z_score,
               calculate_map_histogram]
    for m in methods:
        heatmap = get_heatmap(img, m)
        cv2.imwrite(str(m) + '_heatmap.png', heatmap)


def outliers(img, broken_px, percents):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for name, method in methods.items():
        potential_px = get_outliers(img, method['method'], percents)
        if broken_px in potential_px:
            methods[name]['score'] += 1


if __name__ == '__main__':
    # interpolate.main()          # run interpolate funcs to calculate statistic

    img_name_list = os.listdir(broken_dataset_path)
    total_img = len(img_name_list)
    print(f"total_img_count: {total_img}")
    for i in range(80, 100):

        for img in img_name_list:
            broken_img = cv2.imread(broken_dataset_path + img)
            broken_px = find_diff_px(img)
            outliers(broken_img, broken_px, i)

        print(f"i = {i}\n"
              f"{[{'name': name, 'score': round(method['score']/total_img, 4)} for name, method in methods.items()]}")

        methods = {"mehalanobis": {"score": 0, "method": calculate_map_mehalanobis_distance},
                   "outlier_detection": {"score": 0, "method": calculate_map_outlier_detection},
                   "z_score": {"score": 0, "method": calculate_map_z_score},
                   "histogram": {"score": 0, "method": calculate_map_histogram}}

    # img_name = "11774.png"
    # img_name = "12195.png"
    # img_name = "159193.png"
    # broken_img = cv2.imread(broken_dataset_path + img_name)
    # origin_img = cv2.imread(origin_dataset_path + img_name)
    #
    # # # heatmap(img=im)
    # #
    # cv2.imshow("lol", broken_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # mean = find_min_pixels_count(outlier_detector.z_score, broken_dataset_path + img_name, 0, 100)
    # # print(mean)
    # # res = outlier_detector.z_score(broken_dataset_path + img_name, mean)
    # # print(res)
