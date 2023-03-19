import cv2
import outlier_detector

from utils.img_worker import find_diff_px
from configurations import broken_dataset_path

def find_min_pixels_count(method, data, m1, m2, max_iterations = 15):
    res_m1 = method(data, m1)
    res_m2 = method(data, m2)
    iter = 0
    res = res_m1
    while len(res) != 1 and iter < max_iterations:
        mean = (m1 + m2)/2
        res_mean = method(data, mean)
        if len(res_mean) > 1:
            m1 = mean
        elif len(res_mean) == 0:
            m2 = mean
        res = res_m1
        iter+=1
    return mean

if __name__ == '__main__':
    img_name = "1335.png"
    # img_name = "11774.png"
    # img_name = "12195.png"
    im = cv2.imread(broken_dataset_path+img_name)
    # cv2.imshow("lol", im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    mean = find_min_pixels_count(outlier_detector.z_score, broken_dataset_path+img_name, 0, 100)
    print(mean)
    res = outlier_detector.z_score(broken_dataset_path+img_name, mean)
    print(res)
    print(find_diff_px(img_name))
