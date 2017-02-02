
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras import backend as K


smooth = 1e-12


def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    union = K.sum(y_true + y_pred, axis=[0, -1, -2]) - intersection

    return K.mean(intersection / (union + smooth))


def jaccard_coef_int(y_true, y_pred):
    return jaccard_coef(y_true, K.round(K.clip(y_pred, 0, 1)))


def combined_loss(y_true, y_pred):
    return K.binary_crossentropy(y_pred, y_true) + 0.2 * (1 - jaccard_coef(y_true, y_pred))


def unet(input_shape, n_classes):
    inputs = Input(input_shape)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)

    up7 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(n_classes, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(), loss=combined_loss, metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])

    return model


def unet2(input_shape, n_classes):
    def activation():
        return Activation('relu')

    def conv_block(x, n_units, factorize=True):
        if factorize:
            x = Convolution2D(n_units, 3, 1, border_mode='same', init='he_normal')(x)
            x = Convolution2D(n_units, 1, 3, border_mode='same', init='he_normal')(x)
        else:
            x = Convolution2D(n_units, 3, 3, border_mode='same', init='he_normal')(x)

        x = activation()(x)

        if factorize:
            x = Convolution2D(n_units, 3, 1, border_mode='same', init='he_normal')(x)
            x = Convolution2D(n_units, 1, 3, border_mode='same', init='he_normal')(x)
        else:
            x = Convolution2D(n_units, 3, 3, border_mode='same', init='he_normal')(x)

        x = activation()(x)

        return x

    def pool_block(x, pool_size):
        x = MaxPooling2D((pool_size, pool_size))(x)
        #x = BatchNormalization(axis=1)(x)
        #x = Dropout(0.5)(x)
        return x

    def merge_block(conv, skip):
        return merge([UpSampling2D(size=(2, 2))(conv), skip], mode='concat', concat_axis=1)

    inputs = Input(input_shape)

    conv1 = conv_block(inputs, 48, factorize=False)
    pool1 = pool_block(conv1, 2)
    #pool1 = Dropout(0.1)(pool1)

    conv2 = conv_block(pool1, 64)
    pool2 = pool_block(conv2, 2)
    #pool2 = Dropout(0.2)(pool2)

    conv3 = conv_block(pool2, 96)
    pool3 = pool_block(conv3, 2)
    #pool3 = Dropout(0.3)(pool3)

    conv4 = conv_block(pool3, 128)
    #conv4 = Dropout(0.4)(conv4)

    up7 = merge_block(conv4, conv3)
    conv7 = conv_block(up7, 96)
    #conv7 = Dropout(0.3)(conv7)

    up8 = merge_block(conv7, conv2)
    conv8 = conv_block(up8, 64)
    #conv8 = Dropout(0.2)(conv8)

    up9 = merge_block(conv8, conv1)
    conv9 = conv_block(up9, 48, factorize=False)
    #conv9 = Dropout(0.1)(conv9)

    conv10 = Convolution2D(n_classes, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(), loss=combined_loss, metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])

    return model
