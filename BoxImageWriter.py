import numpy as np
import cv2
import os
from utils import make_boxed_image


class BoxImageWriter:
    """
    Writing bounding_boxs onto the original image
    """
    def __init__(self, dir_path: str):
        self.dir_path = dir_path

    def write(self, image: np.ndarray, image_name: str, bboxes):
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)
        box_image = make_boxed_image(image, bboxes)
        box_image_path = os.path.join(self.dir_path, os.path.basename(image_name))
        cv2.imwrite(box_image_path, box_image)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)
