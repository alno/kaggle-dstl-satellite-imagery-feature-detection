import numpy as np

import os

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers.advanced_activations import ELU
from keras import backend as K

from util.meta import image_size, mask_size, n_classes, val_train_image_ids, val_test_image_ids


n_patches = 16

image_patch_size = image_size / n_patches
mask_patch_size = mask_size / n_patches

smooth = 1e-12


def normalize(x):
    for c in xrange(x.shape[0]):
        q5, q95 = np.percentile(x[c], [5, 95])

        x[c] = np.clip((x[c] - q5) / (q95 - q5), 0, 1)

    return x


def load_labelled_patches(image_ids):
    print "Loading train patches..."

    n = len(image_ids) * n_patches * n_patches

    xx = np.zeros((n, 3, image_patch_size, image_patch_size), dtype=np.float32)
    yy = np.zeros((n, n_classes, mask_patch_size, mask_patch_size), dtype=np.uint8)

    k = 0
    for image_id in image_ids:
        x = normalize(np.load('cache/images/%s.npy' % image_id))
        y = np.load('cache/masks/%s.npy' % image_id)

        for i in xrange(n_patches):
            for j in xrange(n_patches):
                xx[k] = x[:, i*image_patch_size:(i+1)*image_patch_size, j*image_patch_size:(j+1)*image_patch_size]
                yy[k] = y[:, i*mask_patch_size:(i+1)*mask_patch_size, j*mask_patch_size:(j+1)*mask_patch_size]
                k += 1

    return xx, yy


def train_batch_generator(x, y, batch_size=64):
    index = np.arange(x.shape[0])

    while True:
        np.random.shuffle(index)

        batch_start = 0
        while batch_start < x.shape[0]:
            batch_index = index[batch_start:batch_start + batch_size]
            batch_start += batch_size

            x_batch = x[batch_index].copy()
            y_batch = y[batch_index].copy()

            for i in xrange(x_batch.shape[0]):
                if np.random.random() < 0.5:  # Mirror by x
                    x_batch[i] = x_batch[i, :, ::-1, :]
                    y_batch[i] = y_batch[i, :, ::-1, :]

                if np.random.random() < 0.5:  # Mirror by y
                    x_batch[i] = x_batch[i, :, :, ::-1]
                    y_batch[i] = y_batch[i, :, :, ::-1]

                if np.random.random() < 0.5:  # Rotate
                    x_batch[i] = np.swapaxes(x_batch[i], 1, 2)
                    y_batch[i] = np.swapaxes(y_batch[i], 1, 2)

            yield x_batch, y_batch


def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def combined_loss(y_true, y_pred):
    return K.binary_crossentropy(y_pred, y_true) + 0.2 * (1 - jaccard_coef(y_true, y_pred))


def get_unet():
    inputs = Input((3, image_patch_size, image_patch_size))
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(inputs)
    conv1 = Activation('relu')(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(pool1)
    conv2 = Activation('relu')(conv2)
    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(pool2)
    conv3 = Activation('relu')(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(pool3)
    conv4 = Activation('relu')(conv4)
    conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, border_mode='same', init='he_uniform')(pool4)
    conv5 = Activation('relu')(conv5)
    conv5 = Convolution2D(512, 3, 3, border_mode='same', init='he_uniform')(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(up6)
    conv6 = Activation('relu')(conv6)
    conv6 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(up7)
    conv7 = Activation('relu')(conv7)
    conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(up8)
    conv8 = Activation('relu')(conv8)
    conv8 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(up9)
    conv9 = Activation('relu')(conv9)
    conv9 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv9)
    conv9 = Activation('relu')(conv9)

    conv10 = Convolution2D(n_classes, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(), loss=combined_loss, metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    return model


def train_model(x_train, y_train, x_val, y_val):
    print "Defining model..."

    model = get_unet()
    model_checkpoint = ModelCheckpoint('cache/unet_tmp.hdf5', monitor='loss', save_best_only=True)

    print "Training model..."

    model.fit_generator(
        train_batch_generator(x_train, y_train, batch_size=64),
        samples_per_epoch=x_train.shape[0],
        nb_epoch=50, verbose=1,
        callbacks=[model_checkpoint], validation_data=(x_val, y_val))

    return model


# Validation pass
if True:
    train_x, train_y = load_labelled_patches(val_train_image_ids)
    val_x, val_y = load_labelled_patches(val_test_image_ids)

    train_model(train_x, train_y, val_x, val_y)
