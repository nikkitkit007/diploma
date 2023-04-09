from typing import NamedTuple, Dict, List
import os
import cv2
import time

from configurations import broken_dataset_path, origin_dataset_path
from utils.img_worker import find_diff_px

from fixers.inpaint import Inpainter
from fixers.interpolate import Interpolator


class Method(Dict):
    name: str
    call: callable
    score: float
    quality: float
    time: float


interpolator_methods = [Method(name="nearest_neighbor", call=Interpolator.nearest_neighbor, score=0, quality=0, time=0),
                        Method(name="bilinear", call=Interpolator.bilinear, score=0, quality=0, time=0),
                        Method(name="bicubic", call=Interpolator.bicubic, score=0, quality=0, time=0),
                        Method(name="lanczos", call=Interpolator.lanczos, score=0, quality=0, time=0),
                        Method(name="custom", call=Interpolator.custom, score=0, quality=0, time=0), ]


inpaint_methods = [Method(name="navier_stokes", call=Inpainter.navier_stokes, score=0, quality=0, time=0),
                   Method(name="telea", call=Inpainter.telea, score=0, quality=0, time=0), ]


avg_proc = {}
proc = {}


def analyze(methods: List[Method], img_names: List[str]):
    for method in methods:
        proc[method['name']] = 0

    for img_name in img_names:
        broken_img = cv2.imread(broken_dataset_path + img_name)
        origin_img = cv2.imread(origin_dataset_path + img_name)

        x, y = find_diff_px(img_name)
        bgr_origin = origin_img[x][y]  # BGR (b, g, r)
        bgr_broken = broken_img[x][y]  # BGR (b, g, r)

        dist = {}
        try:
            for method in methods:
                time_start = time.time()
                method_dist = Interpolator.calc_dist(method['call'](broken_img, x, y), bgr_origin)
                time_finish = time.time() - time_start

                dist[method['name']] = method_dist
                proc[method['name']] += method_dist
                # time[method['name']]
                method['time'] += time_finish

            dist = dict(sorted(dist.items(), key=lambda x: x[1]))
            dist_rating = {}
            for i, d in enumerate(dist):
                dist_rating[d] = i
            for rate, method in enumerate(methods):
                method['score'] += len(methods) - dist_rating[method['name']]
        except:
            pass

    total_img_count = len(img_names)
    for pr in proc:
        proc[pr] = round(1 - (proc[pr] / total_img_count) / 255, 4)

    for method in methods:
        method['quality'] += proc[method['name']]

    for method in methods:
        method['time'] = round(method['time'] / total_img_count, 10)

    for method in methods:
        method['score'] = method['score'] / 1000


def print_top(data: List[Method], key: str, reverse=False):
    print(f"\nTOP {key.upper()}:")
    top = sorted(data, key=lambda x: x[key], reverse=reverse)
    for method in top:
        print(f"{method['name']} - {method[key]}")


if __name__ == '__main__':
    img_names = os.listdir(broken_dataset_path)[:]

    fix_methods = inpaint_methods + interpolator_methods
    analyze(methods=fix_methods, img_names=img_names)

    # -----------------------top score--------------------------
    print_top(data=fix_methods, key='score', reverse=True)

    # ----------------------top quality-------------------------
    print_top(data=fix_methods, key='quality', reverse=True)

    # ------------------------top time--------------------------
    print_top(data=fix_methods, key='time', reverse=False)

