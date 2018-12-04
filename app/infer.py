import pdb
from os.path import join

import cv2
import numpy as np

from model import model_1

MODEL = 'weights-0048-0.75.hdf5'
IMAGE = 'cat.0.jpg'


def infer(model, image_array):
  norm_image_array = np.array(x_batch, np.float32) / 255
  prediction = model.predict(norm_image_array)
  return prediction

model = model_1()
weights_path = join('../models', MODEL)
image_path = join('../data', IMAGE)
image = cv2.imread(image_path)
image = cv2.resize(image, (128, 128))

image_array = np.empty([1, 128, 128, 3])
image_array[0] = image

pdb.set_trace()
model.load_weights(weights_path)
prediction = infer(model, image_array)

