from typing import NamedTuple, Dict, List
import os
import cv2

from configurations import broken_dataset_path, origin_dataset_path
from utils.img_worker import find_diff_px

from fixers.inpaint import Inpainter
from fixers.interpolate import Interpolator


class Method(Dict):
    name: str
    call: callable
    score: float
    quality: float


interpolator_methods = [Method(name="nearest_neighbor", call=Interpolator.nearest_neighbor, score=0, quality=0),
                        Method(name="bilinear", call=Interpolator.bilinear, score=0, quality=0),
                        Method(name="bicubic", call=Interpolator.bicubic, score=0, quality=0),
                        Method(name="lanczos", call=Interpolator.lanczos, score=0, quality=0),
                        Method(name="custom", call=Interpolator.custom, score=0, quality=0), ]


inpaint_methods = [Method(name="navier_stokes", call=Inpainter.navier_stokes, score=0, quality=0),
                   Method(name="telea", call=Inpainter.telea, score=0, quality=0), ]


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
                method_dist = Interpolator.calc_dist(method['call'](broken_img, x, y), bgr_origin)
                dist[method['name']] = method_dist
                proc[method['name']] += method_dist

            dist = dict(sorted(dist.items(), key=lambda x: x[1]))
            dist_rating = {}
            for i, d in enumerate(dist):
                dist_rating[d] = i
            for rate, method in enumerate(methods):
                method['score'] += dist_rating[method['name']]
        except:
            pass

    total_img_count = len(img_names)
    for pr in proc:
        proc[pr] = round(1 - (proc[pr] / total_img_count) / 255, 4)

    for method in methods:
        method['quality'] += proc[method['name']]


if __name__ == '__main__':
    img_names = os.listdir(broken_dataset_path)[:]

    fix_methods = inpaint_methods + interpolator_methods
    analyze(methods=fix_methods, img_names=img_names)

    # -----------------------top score--------------------------
    print("\nTOP SCORE")
    top_score = sorted(fix_methods, key=lambda x: x['score'], reverse=True)
    for method in top_score:
        print(f"{method['name']} - {method['score']}")

    # ----------------------top quality-------------------------
    print("\nTOP QUALITY")
    top_quality = sorted(fix_methods, key=lambda x: x['quality'], reverse=True)
    for method in top_quality:
        print(f"{method['name']} - {method['quality']}")
