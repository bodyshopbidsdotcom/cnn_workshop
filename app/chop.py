import cv2
from os.path import join
from keras.models import Model
import numpy as np
import pdb
from imutils import build_montages

from model import model_1, model_2, model_alexnet

IMAGE = 'cat.0.jpg'


def build_bottleneck_model(model, layer_name):
    for layer in model.layers:
        if layer.name == layer_name:
            output = layer.output

    bottleneck_model = Model(model.input, output)
    bottleneck_model.compile(loss='categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])

    return bottleneck_model


alex, name = model_alexnet(input_shape=(128, 128, 3))
alex.summary()
alex.load_weights('weights-0032-0.92.hdf5')
target_layer = 'conv2d_1'

model = build_bottleneck_model(alex, target_layer)

image_path = join('data', IMAGE)
image = cv2.imread(image_path)
image = np.array(image, np.float32) / 255
image = cv2.resize(image, (128, 128))
image = np.expand_dims(image, axis=0)

predictions = model.predict(image)

target_shape = (128, 128)
#target_shape = (64, 64)
num_predictions = predictions.shape[3]

predictions = predictions * 255
predictions = predictions.astype('uint8')
predictions = np.dsplit(predictions[0], num_predictions)  # Shape is (1, <width>, <height>, <depth>). Need to remove the leading 1.
zero_stack = np.zeros(target_shape)
predictions = [np.dstack((np.squeeze(prediction), zero_stack, zero_stack)) for prediction in predictions]
montages = build_montages(predictions, target_shape, (7, 7))

for montage in montages:
    cv2.imshow('Filters', montage)
    cv2.waitKey()
