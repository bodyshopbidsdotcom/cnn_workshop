import keras.callbacks as keras_callbacks
import sklearn.model_selection as sk_model_selection
from os import listdir, mkdir, rmdir
from os.path import join, isdir
import numpy as np
from random import shuffle
import pdb
import cv2
from uuid import uuid4

from model import model_1, model_2, model_alexnet, model_inception_v3
from perturbations import perturb
from constants import DATA, BATCH_SIZE, TEST_SIZE, RANDOM_SEED, MODELS_FOLDER, EPOCHS, IMAGE_FILE_SUFFIXES


def create_model_folder(folder_name):
    if isdir(folder_name):
        rmdir(folder_name)
    mkdir(folder_name)


def classification_vector(filename):
    rv = np.zeros(2)
    if 'cat' in filename:
        rv[0] = 1
    else:
        rv[1] = 1
    return rv


def generator(files, image_directory, perturb_function=None):
    while True:
        for start in range(0, len(files), BATCH_SIZE):
            x_batch = []
            y_batch = []
            end = min(start + BATCH_SIZE, len(files))
            ids_train_batch = files[start:end]
            for id in ids_train_batch:
                img = cv2.imread(join(image_directory, id))

                if perturb_function is not None:
                    img = perturb_function(img)

                x_batch.append(img)
                y_batch.append(classification_vector(id))

            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32)

            yield x_batch, y_batch


def train(model, model_name):
    models_folder = join(MODELS_FOLDER, model_name)
    create_model_folder(models_folder)

    files = listdir(DATA)
    files = [file for file in files if file.lower().endswith(tuple(IMAGE_FILE_SUFFIXES))]
    shuffle(files)
    train_files, valid_files = sk_model_selection.train_test_split(files, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    callbacks = [keras_callbacks.EarlyStopping(monitor='val_loss',
                                               patience=8,
                                               verbose=1,
                                               min_delta=1e-4),
                 keras_callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                   factor=0.1,
                                                   patience=4,
                                                   verbose=1,
                                                   epsilon=1e-4),
                 keras_callbacks.ModelCheckpoint(monitor='val_loss',
                                                 filepath=join(models_folder,
                                                               'weights-{epoch:04d}-{val_acc:.2f}.hdf5'),
                                                 save_best_only=True,
                                                 save_weights_only=False)]

    model.fit_generator(
        generator(train_files, DATA, perturb),
        steps_per_epoch=np.ceil(float(len(train_files)) / float(BATCH_SIZE)),
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks,
        validation_data=generator(valid_files, DATA),
        validation_steps=np.ceil(float(len(valid_files)) / float(BATCH_SIZE)))


model, name = model_inception_v3(input_shape=(1024, 1024, 3), classes=2)
train(model, name)


'''
model, name = model_1()
train(model, name)
model, name = model_1(number_of_blocks=6)
train(model, name)
model, name = model_1(number_of_blocks=9)
train(model, name)
model, name = model_2()
train(model, name)
'''
