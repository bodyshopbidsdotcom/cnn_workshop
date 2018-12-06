from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, Dense, Flatten, Dropout
from keras.losses import mean_squared_error
from keras.metrics import binary_accuracy
from keras.optimizers import RMSprop


def model_1(input_shape=(128, 128, 3), num_classes=2, kernel_size=3, number_of_blocks=3):
    '''
        Basic model consisting of convolution, activation, and max pooling stacked number_of_block times.
    '''
    inputs = Input(shape=input_shape)
    prev_layer = inputs

    filters = 64

    for i in range(number_of_blocks):
        conv = Conv2D(filters, (kernel_size, kernel_size), padding='same')(prev_layer)
        acti = Activation('relu')(conv)
        maxp = MaxPooling2D((2, 2), strides=(8, 8))(acti)

        prev_layer = maxp
        filters *= 2

    flatten = Flatten()(prev_layer)
    dense = Dense(num_classes, activation='sigmoid')(flatten)
    bn = BatchNormalization()(dense)
    act = Activation()(bn)
    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=RMSprop(lr=0.0001), loss=mean_squared_error, metrics=[binary_accuracy])

    return model, 'model_1_{}_{}'.format(num_classes, number_of_blocks)


def model_2(input_shape=(128, 128, 3), num_classes=2, kernel_size=3, number_of_blocks=3):
    '''
        Model consisting of convolution, batch normalization, activation, and max pooling stacked number_of_block times.
    '''
    inputs = Input(shape=input_shape)
    prev_layer = inputs

    filters = 64

    for i in range(number_of_blocks):
        conv = Conv2D(filters, (kernel_size, kernel_size), padding='same')(prev_layer)
        bn = BatchNormalization()(conv)
        acti = Activation('relu')(bn)
        maxp = MaxPooling2D((2, 2), strides=(8, 8))(acti)

        prev_layer = maxp
        filters *= 2

    flatten = Flatten()(prev_layer)
    dense = Dense(num_classes, activation='sigmoid')(flatten)
    bn = BatchNormalization()(dense)
    act = Activation()(bn)
    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=RMSprop(lr=0.0001), loss=mean_squared_error, metrics=[binary_accuracy])

    return model, 'model_1_{}_{}'.format(num_classes, number_of_blocks)


def model_alexnet(input_shape=(256, 256, 3), num_classes=2):
    inputs = Input(shape=input_shape)

    # Layer 1
    conv = Conv2D(96, (11, 11), padding='same')(inputs)
    bn = BatchNormalization()(conv)
    act = Activation('relu')(bn)
    maxp = MaxPooling2D((2, 2))(act)  # 256 -> 128

    # Layer 2
    conv = Conv2D(256, (5, 5), padding='same')(maxp)
    bn = BatchNormalization()(conv)
    act = Activation('relu')(bn)
    maxp = MaxPooling2D((2, 2))(act)  # 128 -> 64

    # Layer 3
    # Missing ZeroPadding2d
    conv = Conv2D(512, (3, 3), padding='same')(maxp)
    bn = BatchNormalization()(conv)
    act = Activation('relu')(bn)
    maxp = MaxPooling2D((2, 2))(act)  # 64 -> 32

    # Layer 4
    # Missing ZeroPadding2d
    conv = Conv2D(1024, (3, 3), padding='same')(maxp)
    bn = BatchNormalization()(conv)
    act = Activation('relu')(bn)

    # Layer 5
    # Missing ZeroPadding2d
    conv = Conv2D(1024, (3, 3), padding='same')(maxp)
    bn = BatchNormalization()(conv)
    act = Activation('relu')(bn)
    maxp = MaxPooling2D((2, 2))(act)  # 32 -> 16

    # Layer 6
    flatten = Flatten()(maxp)
    dense = Dense(3072)(flatten)
    bn = BatchNormalization()(dense)
    act = Activation('relu')(bn)
    drop = Dropout(0.5)(act)

    # Layer 7
    dense = Dense(4096)(drop)
    bn = BatchNormalization()(dense)
    act = Activation('relu')(bn)
    drop = Dropout(0.5)(act)

    # Layer 8
    dense = Dense(num_classes)(drop)
    bn = BatchNormalization()(dense)
    act = Activation('softmax')(bn)

    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=RMSprop(lr=0.0001), loss=mean_squared_error, metrics=[binary_accuracy])

    return model, 'model_alexnet'
