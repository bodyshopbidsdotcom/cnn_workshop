import keras.callbacks as keras_callbacks
import sklearn.model_selection as sk_model_selection
from os import listdir
from os.path import join
import numpy as np
import pdb
import cv2

from model import model_1
from perturbations import perturb
from constants import DATA, BATCH_SIZE, TEST_SIZE, RANDOM_SEED, MODELS_FOLDER, EPOCHS


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


def train():

    files = listdir(DATA)
    train_files, valid_files = sk_model_selection.train_test_split(files, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    basic_model = model_1()

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
                                                 filepath=join(MODELS_FOLDER,
                                                               'weights-{epoch:04d}-{val_dice_coeff:.2f}.hdf5'),
                                                 save_best_only=True,
                                                 save_weights_only=True)]

    basic_model.fit_generator(
        generator(train_files, DATA, perturb),
        steps_per_epoch=np.ceil(float(len(train_files)) / float(BATCH_SIZE)),
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks,
        validation_data=generator(valid_files, DATA),
        validation_steps=np.ceil(float(len(valid_files)) / float(BATCH_SIZE)))


train()
