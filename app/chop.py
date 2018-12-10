import cv2
from os import listdir
from os.path import join
from keras.models import Model
import numpy as np
import pdb
from imutils import build_montages
from sklearn.cluster import KMeans

from model import model_1, model_2, model_alexnet

#IMAGE = 'cat.93.jpg'
#IMAGE = 'slovenia.png'
#IMAGE = 'michael.png'
INPUT_SHAPE = (128, 128)
PREDICTED_SHAPE = (8, 8)


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


alex, name = model_alexnet(input_shape=(128, 128, 3))
alex.summary()
alex.load_weights('weights-0032-0.92.hdf5')
target_layer = 'max_pooling2d_4'

model = build_bottleneck_model(alex, target_layer)

features = []
count = 0
images = listdir('data')
for image_filename in images:
    image_path = join('data', image_filename)

    predictions = generate_prediction(image_path, model)
    pdb.set_trace()
    print('Predicted {}'.format(count))
    count += 1
    features.append(predictions)

'''
montage_shape = (128, 128)
montages = build_montages(predictions, montage_shape, (7, 7))

for montage in montages:
    cv2.imshow(IMAGE, montage)
    if cv2.waitKey() == 113:
        break
'''

