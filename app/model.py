from losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff


from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.metrics import binary_accuracy
from keras.optimizers import RMSprop


def model_inception_v3(input_shape, classes):
    base_model = InceptionV3(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return model, 'inception_v3'


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
    act = Activation('relu')(bn)
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
    act = Activation('relu')(bn)
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

    model = Model(inputs=inputs, outputs=act)  # output should be act
    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])

    return model, 'model_alexnet'


def model_unet_1024(input_shape=(1024, 1024, 3),
                    num_classes=1, kernel_size=3):
    inputs = Input(shape=input_shape)
    # 1024

    down0b = Conv2D(8, (3, 3), padding='same')(inputs)
    down0b = BatchNormalization()(down0b)
    down0b = Activation('relu')(down0b)
    down0b = Conv2D(8, (3, 3), padding='same')(down0b)
    down0b = BatchNormalization()(down0b)
    down0b = Activation('relu')(down0b)
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    # 512

    down0a = Conv2D(16, (3, 3), padding='same')(down0b_pool)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    # 512

    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    # 1024

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0b)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model, 'unet_1024'

