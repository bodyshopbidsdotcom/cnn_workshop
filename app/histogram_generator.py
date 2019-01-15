from os import listdir, mkdir
from os.path import join, isdir
import pdb
import cv2
import numpy as np
import pickle
from sklearn.cluster import KMeans

from constants import IMAGE_FILE_SUFFIXES, CHANNELS, DATA, CLUSTERS_DATA


def print_cluster_quality():
    pdb.set_trace()
    folders = listdir(CLUSTERS_DATA)
    folders = [file for file in folders if isdir(join(CLUSTERS_DATA, file))]

    for folder in folders:
        files = listdir(join(CLUSTERS_DATA, folder))

        counts = { 'yes': 0, 'no': 0}
        for file in files:
            if 'yes' in file:
                counts['yes'] += 1
            else:
                counts['no'] += 1

        print('folder: {}'.format(folder))
        print('yes: {} ({})', counts['yes'], counts['yes'] / len(files))
        print('no:  {} ({})', counts['no'], counts['no'] / len(files))


def get_image(file):
    image = cv2.imread(file)

    adj_channels = []
    channels = cv2.split(image)
    for channel in channels:
        mean = np.mean(channel)
        adj_channel = np.subtract(channel, mean)
        adj_channels.append(adj_channel)

    return cv2.merge(adj_channels)


def mass_predict(model):
    files = listdir(DATA)
    files = [file for file in files if file.lower().endswith(tuple(IMAGE_FILE_SUFFIXES))]

    count = 0
    for file in files:
        image_path = join(DATA, file)
        image = get_image(image_path)
        histogram, bins = np.histogram(image.ravel(), 3 * 256, [-256, 256])
        cluster = str(model.predict([histogram])[0])

        cluster_path = join(CLUSTERS_DATA, cluster)
        if not isdir(cluster_path):
            mkdir(cluster_path)

        new_image_path = join(cluster_path, file)
        cv2.imwrite(new_image_path, image)

        print(count)
        count += 1


def kmeans_histograms(histograms, no_clusters, filename):
    kmeans = KMeans(n_clusters=no_clusters, random_state=0).fit(histograms)
    pickle.dump(kmeans, open(filename, 'wb'))


def generate_histograms():
    histograms = []
    files = listdir(DATA)
    files = [file for file in files if file.lower().endswith(tuple(IMAGE_FILE_SUFFIXES))]

    count = 0
    for file in files:
        image_path = join(DATA, file)
        image = get_image(image_path)

        histogram, bins = np.histogram(image.ravel(), 3 * 256, [-256, 256])
        histograms.append(histogram)

        print(count)
        count += 1

    return histograms

'''
# Generate Clusters
filename = 'kmeans_histogram.pickle'
histograms = generate_histograms()
kmeans_histograms(histograms, 16, filename)


# Predict
model = pickle.load(open(filename, 'rb'))
mass_predict(model)
'''

print_cluster_quality()
