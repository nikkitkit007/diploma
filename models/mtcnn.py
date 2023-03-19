import time

from mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt


class MTCNN_wrapper(object):
    def __init__(self):
        self.model = MTCNN()
        self.name = "MTCNN"
        self.input_size = (32, 32, 3)
        self.num_classes = 2

    def preprocess_input(self, img: list):
        MTCNN_wrapper._show(img)
        # img = cv2.imread('images/img_align_celeba/000001.jpg ', 1)
        image = cv2.resize(img, (self.input_size[0], self.input_size[1]))
        MTCNN_wrapper._show(image)
        return image

    def predict(self, image: list, top=None):
        if top is None:
            top = self.num_classes
        image = self.preprocess_input(image)
        predictions = self.model.detect_faces(image)
        if not predictions:
            return 0
        confidence = predictions[0]['confidence']
        print(confidence)
        return confidence

    @staticmethod
    def _show(image):
        plt.imshow(image)
        plt.show()
