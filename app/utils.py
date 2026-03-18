import cv2
import numpy as np
import os

MIN_WIDTH = 1280
MIN_HEIGHT = 720
BLUR_THRESHOLD = 80


def create_project_structure(base_path):

    os.makedirs(os.path.join(base_path, "raw"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "processed"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "output"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "logs"), exist_ok=True)


def validate_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return False

    height, width = image.shape[:2]

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur < BLUR_THRESHOLD:
        return False

    brightness = np.mean(gray)

    if brightness < 30 or brightness > 230:
        return False

    return True