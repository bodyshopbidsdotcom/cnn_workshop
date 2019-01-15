from os import listdir
from os.path import path

import cv2

from constants import DATA, SHAPE


def resize():
    files = listdir(DATA)
    for file in files:
        path = join(DATA, file)
        image = cv2.imread(path)
        image = cv2.resize(image, SHAPE)
        cv2.imwrite(path, image)


resize()
