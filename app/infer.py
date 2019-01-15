import pdb
from os import listdir
from os.path import join

import cv2
import numpy as np

from model import model_alexnet, model_inception_v3
from constants import IMAGE_FILE_SUFFIXES


MODEL = 'weights-0016-0.36.hdf5'


def infer(model, image_array):
    norm_image_array = np.array(image_array, np.float32) / 255
    prediction = model.predict(norm_image_array)
    return prediction



model, name = model_inception_v3(input_shape=(256, 256, 3))
model.load_weights(MODEL)

folder = '/Users/jaskiemr/Downloads'
files = listdir(folder)
files = [file for file in files if file.lower().endswith(tuple(IMAGE_FILE_SUFFIXES))]

for file in files:
  image_path = join(folder, file)
  image = cv2.imread(image_path)

  image_array = np.empty([1, 256, 256, 3])
  image_array[0] = image

  pdb.set_trace()
  prediction = infer(model, image_array)

  print(prediction)
