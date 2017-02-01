import numpy as np

import datetime
import time
import cv2

import shapely.wkt

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Activation, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras import backend as K

from util.meta import n_classes, val_train_image_ids, val_test_image_ids, full_train_image_ids
from util.data import grid_sizes, sample_submission
from util.masks import mask_to_poly

from math import ceil


patch_size = 64
downscale = 3

n_patches = int(ceil(3400.0 / patch_size / downscale))

smooth = 1e-12


def load_labelled_patches(image_ids):
    print "Loading image patches..."

    n = len(image_ids) * n_patches * n_patches

    xx = np.zeros((n, 3, patch_size, patch_size), dtype=np.float16)
    yy = np.zeros((n, n_classes, patch_size, patch_size), dtype=np.uint8)

    k = 0
    for image_id in image_ids:
        x = np.load('cache/images/%s.npy' % image_id)
        y = np.load('cache/masks/%s.npy' % image_id)

        i_patch_step = (x.shape[1] - patch_size*downscale) / (n_patches - 1.0)
        j_patch_step = (x.shape[2] - patch_size*downscale) / (n_patches - 1.0)

        for i in xrange(n_patches):
            for j in xrange(n_patches):
                si = int(round(i*i_patch_step))
                sj = int(round(j*j_patch_step))

                if downscale == 1:
                    xx[k] = x[:, si:si+patch_size, sj:sj+patch_size]
                    yy[k] = y[:, si:si+patch_size, sj:sj+patch_size]
                else:
                    for c in xrange(xx.shape[1]):
                        xx[k, c] = cv2.resize(x[c, si:si+patch_size*downscale, sj:sj+patch_size*downscale].astype(np.float32), (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

                    for c in xrange(yy.shape[1]):
                        yy[k, c] = cv2.resize(y[c, si:si+patch_size*downscale, sj:sj+patch_size*downscale], (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

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
    def activation():
        return LeakyReLU()

    def pooling(pool_size):
        return MaxPooling2D(pool_size)

    inputs = Input((3, patch_size, patch_size))
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(inputs)
    conv1 = activation()(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv1)
    conv1 = activation()(conv1)
    pool1 = pooling((2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(pool1)
    conv2 = activation()(conv2)
    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv2)
    conv2 = activation()(conv2)
    pool2 = pooling((2, 2))(conv2)
    #pool2 = Dropout(0.1)(pool2)

    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(pool2)
    conv3 = activation()(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv3)
    conv3 = activation()(conv3)
    pool3 = pooling((2, 2))(conv3)
    #pool3 = Dropout(0.1)(pool3)

    conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(pool3)
    conv4 = activation()(conv4)
    conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(conv4)
    conv4 = activation()(conv4)
    #conv4 = Dropout(0.1)(conv4)

    up7 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(up7)
    conv7 = activation()(conv7)
    conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv7)
    conv7 = activation()(conv7)
    #conv7 = Dropout(0.1)(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(96, 3, 3, border_mode='same', init='he_uniform')(up8)
    conv8 = activation()(conv8)
    conv8 = Convolution2D(96, 3, 3, border_mode='same', init='he_uniform')(conv8)
    conv8 = activation()(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(up9)
    conv9 = activation()(conv9)
    conv9 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv9)
    conv9 = activation()(conv9)

    conv10 = Convolution2D(n_classes, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(), loss=combined_loss, metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    return model


def train_model(x_train, y_train, x_val=None, y_val=None):
    print "Defining model..."

    model = get_unet()
    model_checkpoint = ModelCheckpoint('cache/unet_tmp.hdf5', monitor='loss', save_best_only=True)

    print "Training model with %d params..." % model.count_params()

    #model.load_weights('cache/unet_tmp.hdf5')

    model.fit_generator(
        train_batch_generator(x_train, y_train, batch_size=64),
        samples_per_epoch=x_train.shape[0],
        nb_epoch=40, verbose=1,
        callbacks=[model_checkpoint], validation_data=None if x_val is None else (x_val, y_val))

    return model


def upscale(m):
    res = np.zeros((m.shape[0], m.shape[1]*downscale, m.shape[2]*downscale), dtype=np.float32)

    for c in xrange(m.shape[0]):
        res[c] = cv2.resize(m[c], (m.shape[1]*downscale, m.shape[2]*downscale), interpolation=cv2.INTER_CUBIC)

    return res


def predict_image(model, image_id, save=True):
    start_time = time.time()

    x = np.load('cache/images/%s.npy' % image_id)

    i_patch_step = (x.shape[1] - patch_size*downscale) / (n_patches - 1.0)
    j_patch_step = (x.shape[2] - patch_size*downscale) / (n_patches - 1.0)
    xb = np.zeros((n_patches * n_patches, x.shape[0], patch_size, patch_size), dtype=np.float16)

    k = 0
    for i in xrange(n_patches):
        for j in xrange(n_patches):
            si = int(round(i*i_patch_step))
            sj = int(round(j*j_patch_step))

            xb[k] = x[np.newaxis, :, si:si+patch_size*downscale:downscale, sj:sj+patch_size*downscale:downscale]

            k += 1

    pb = model.predict(xb)

    p = np.zeros((n_classes, x.shape[1], x.shape[2]), dtype=np.float32)
    c = np.zeros((n_classes, x.shape[1], x.shape[2]), dtype=np.float32)

    k = 0
    for i in xrange(n_patches):
        for j in xrange(n_patches):
            si = int(round(i*i_patch_step))
            sj = int(round(j*j_patch_step))

            p[:, si:si+patch_size*downscale, sj:sj+patch_size*downscale] += pb[k] if downscale == 1 else upscale(pb[k])
            c[:, si:si+patch_size*downscale, sj:sj+patch_size*downscale] += 1

            k += 1

    print "    Image %s predicted in %d seconds, %d, %.3f" % (image_id, time.time() - start_time, c.min(), c.mean())

    return p / c


# Validation pass
if True:
    train_x, train_y = load_labelled_patches(val_train_image_ids)
    val_x, val_y = load_labelled_patches(val_test_image_ids)

    model = train_model(train_x, train_y, val_x, val_y)

    for image_id in val_test_image_ids:
        p = predict_image(model, image_id)
        np.save('cache/preds/%s.npy' % image_id, p)

# Full pass

model = train_model(*load_labelled_patches(full_train_image_ids))
subm = sample_submission.copy()


for i in subm.index:
    image_id = subm.loc[i, 'ImageId']
    xymax = (grid_sizes.loc[image_id, 'xmax'], grid_sizes.loc[image_id, 'ymin'])
    cls = subm.loc[i, 'ClassType']

    print "  Processing %s / %d..." % (image_id, cls)

    mask = predict_image(model, image_id)

    subm.loc[i, 'MultipolygonWKT'] = shapely.wkt.dumps(mask_to_poly(mask[cls - 1], xymax))

print "Saving..."

subm.to_csv('subm/subm-%s.csv.gz' % datetime.datetime.now().strftime('%Y%m%d-%H%M'), index=False, compression='gzip')

print "Done."
