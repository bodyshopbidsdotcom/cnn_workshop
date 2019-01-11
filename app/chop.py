import cv2
from os import listdir
from os.path import join
from keras.models import Model
import numpy as np
import pdb
from imutils import build_montages
from sklearn.cluster import KMeans

from model import model_1, model_2, model_alexnet, model_unet_1024

#IMAGE = 'cat.93.jpg'
#IMAGE = 'slovenia.png'
IMAGE = '6721757.png'
INPUT_SHAPE = (1024, 1024)
#PREDICTED_SHAPE = (128, 128)
#PREDICTED_SHAPE = (256, 256)
#PREDICTED_SHAPE = (512, 512)
PREDICTED_SHAPE = (1024, 1024)
TARGET_LAYER = 'activation_2'


def build_bottleneck_model(model, layer_name):
    for layer in model.layers:
        if layer.name == layer_name:
            output = layer.output

    bottleneck_model = Model(model.input, output)
    bottleneck_model.compile(loss='categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])

    return bottleneck_model


def kmeans(input_data):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(input_data)
    labels = kmeans.labels_
    score = metrics.silhouette_score(input_data, labels, metric='euclidean')
    print(score)


def generate_prediction(image_path, model):
    image = cv2.imread(image_path)
    image = np.array(image, np.float32) / 255
    image = cv2.resize(image, INPUT_SHAPE)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)

    num_predictions = predictions.shape[3]

    predictions = predictions * 255
    predictions = predictions.astype('uint8')
    predictions = np.dsplit(predictions[0], num_predictions)  # Shape is (1, <width>, <height>, <depth>). Need to remove the leading 1.
    zero_stack = np.zeros(PREDICTED_SHAPE)
    predictions = [np.dstack((np.squeeze(prediction), zero_stack, zero_stack)) for prediction in predictions]
    return predictions


unet, name = model_unet_1024()
unet.summary()
unet.load_weights('models/114-model-weights.h5')

model = build_bottleneck_model(unet, TARGET_LAYER)


images = listdir('data')
for image in images:
    image_path = join('data', image)
    predictions = generate_prediction(image_path, model)

    montage_shape = (128, 128)
    montages = build_montages(predictions, montage_shape, (7, 6))

    break_out = False
    for montage in montages:
        cv2.imshow('Filters', montage)
        if cv2.waitKey() == 113:
            break_out = True
            break

    if break_out:
        break

