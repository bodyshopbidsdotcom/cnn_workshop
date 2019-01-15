import pdb
from os import listdir
from os.path import join

import cv2
import numpy as np

from model import model_alexnet, model_inception_v3
from constants import IMAGE_FILE_SUFFIXES, TESTING


MODEL = 'weights-0016-0.36.hdf5'


def _infer(model, image_array):
    norm_image_array = np.array(image_array, np.float32) / 255
    prediction = model.predict(norm_image_array)
    return prediction


def confusion_matrix(results):
    '''
                Predicted
                yes  no
    Actual yes  0    0
           no   0    0
    '''

    matrix = np.zeros((2, 2))
    for filename, prediction in results.items():
        if 'yes' in filename:
            if prediction[0] > prediction[1]:
                matrix[0, 0] += 1 # actual yes, predicted yes
            else:
                matrix[1, 0] += 1 # actual yes, predicted no
        else:
            if prediction[0] > prediction[1]:
                matrix[0, 1] += 1 # actual no, predicted yes
            else:
                matrix[1, 1] += 1 # actual no, predicted no


model, name = model_inception_v3(input_shape=(256, 256, 3))
model.load_weights(MODEL)

folder = TESTING
files = listdir(folder)
files = [file for file in files if file.lower().endswith(tuple(IMAGE_FILE_SUFFIXES))]

results = {}

for file in files:
  image_path = join(folder, file)
  image = cv2.imread(image_path)

  image_array = np.empty([1, 256, 256, 3])
  image_array[0] = image

  prediction = infer(model, image_array)
  results['file'] = prediction
  print(prediction)
