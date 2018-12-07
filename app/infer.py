import pdb
from os.path import join

import cv2
import numpy as np

from model import model_alexnet

MODEL = 'weights-0032-0.92.hdf5'
IMAGE = 'cat.0.jpg'


def infer(model, image_array):
    norm_image_array = np.array(image_array, np.float32) / 255
    prediction = model.predict(norm_image_array)
    return prediction


model, name = model_alexnet(input_shape=(128, 128, 3))
model.load_weights(MODEL)

image_path = join('data', IMAGE)
image = cv2.imread(image_path)

image_array = np.empty([1, 128, 128, 3])
image_array[0] = image

pdb.set_trace()
prediction = infer(model, image_array)

print(prediction)
