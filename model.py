from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, Dense, Flatten
from keras.losses import mean_squared_error
from keras.metrics import binary_accuracy
from keras.optimizers import RMSprop


def model_1(input_shape=(128, 128, 3), num_classes=2, kernel_size=3, number_of_blocks=3):
    inputs = Input(shape=input_shape)
    prev_layer = inputs

    filters = 64

    for i in range(number_of_blocks):
        conv = Conv2D(filters, (kernel_size, kernel_size), padding='same')(prev_layer)
        acti = Activation('relu')(conv)
        maxp = MaxPooling2D((2, 2), strides=(2, 2))(acti)

        prev_layer = maxp
        filters *= 2

    flatten = Flatten()(prev_layer)
    classify = Dense(num_classes, activation='sigmoid')(flatten)
    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=RMSprop(lr=0.0001), loss=mean_squared_error, metrics=[binary_accuracy])

    return model


