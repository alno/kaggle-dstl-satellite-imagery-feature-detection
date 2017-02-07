
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Activation, Dropout, BatchNormalization, AtrousConvolution2D
from keras.optimizers import Adam, SGD, Adadelta
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.layers.advanced_activations import ELU, LeakyReLU, PReLU
from keras.regularizers import l2
from keras import backend as K


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[0, -1, -2])
    return K.mean((2. * intersection + smooth) / (K.sum(K.square(y_true), [0, -1, -2]) + K.sum(K.square(y_pred), [0, -1, -2]) + smooth))


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def jaccard_coef(y_true, y_pred, smooth=1e-12):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    union = K.sum(y_true + y_pred, axis=[0, -1, -2]) - intersection

    return K.mean((intersection + smooth) / (union + smooth))


def jaccard_coef_int(y_true, y_pred, smooth=1e-12):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    union = K.sum(y_true + y_pred, axis=[0, -1, -2]) - intersection

    return K.mean((intersection + smooth) / (union + smooth))


def jaccard_coef_loss(y_true, y_pred, smooth=1e-3):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    union = K.sum(K.square(y_true) + K.square(y_pred), axis=[0, -1, -2]) - intersection

    return 1.0 - K.mean((intersection + smooth) / (union + smooth))


def combined_loss(y_true, y_pred):
    return K.binary_crossentropy(y_pred, y_true) + 0.1 * jaccard_coef_loss(y_true, y_pred)


def unet(input_shapes, n_classes):
    inputs = Input(input_shapes['in'], name='in')
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
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(384, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(384, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
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
    model.compile(optimizer=Adam(), loss=combined_loss, metrics=[jaccard_coef, jaccard_coef_int])

    return model


def unet_ma(input_shapes, n_classes):
    in_M = Input(input_shapes['in_M'], name='in_M')
    in_A = Input(input_shapes['in_A'], name='in_A')

    inputs = merge([in_A, in_M], mode='concat', concat_axis=1)

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

    model = Model(input=[in_M, in_A], output=conv10)
    model.compile(optimizer=Adam(), loss=combined_loss, metrics=[jaccard_coef, jaccard_coef_int])

    return model


def conv_block(x, n_units, factorize=False):
    if factorize:
        x = Convolution2D(n_units, 3, 1, border_mode='same', init='he_normal', bias=False)(x)
        x = Convolution2D(n_units, 1, 3, border_mode='same', init='he_normal', bias=False)(x)
    else:
        x = Convolution2D(n_units, 3, 3, border_mode='same', init='he_normal', bias=False)(x)

    x = BatchNormalization(axis=1, mode=0)(x)
    x = PReLU(shared_axes=[2, 3])(x)

    return x


def pool_block(x, pool_size):
    return MaxPooling2D((pool_size, pool_size))(x)


def merge_block(conv, skip, mode='concat'):
    return merge([UpSampling2D(size=(2, 2))(conv), skip], mode=mode, concat_axis=1)


def unet2(input_shapes, n_classes):
    inputs = Input(input_shapes['in'], name='in')

    conv1 = conv_block(inputs, 48)
    conv1 = conv_block(conv1, 48)
    pool1 = pool_block(conv1, 2)

    conv2 = conv_block(pool1, 64)
    conv2 = conv_block(conv2, 64)
    pool2 = pool_block(conv2, 2)

    conv3 = conv_block(pool2, 128)
    conv3 = conv_block(conv3, 128)
    pool3 = pool_block(conv3, 2)

    conv4 = conv_block(pool3, 256)
    conv4 = conv_block(conv4, 256)
    pool4 = pool_block(conv4, 2)

    conv5 = conv_block(pool4, 512)
    conv5 = conv_block(conv5, 512)

    up6 = merge_block(conv5, conv4)
    conv6 = conv_block(up6, 256)
    conv6 = conv_block(conv6, 256)

    up7 = merge_block(conv6, conv3)
    conv7 = conv_block(up7, 128)
    conv7 = conv_block(conv7, 128)

    up8 = merge_block(conv7, conv2)
    conv8 = conv_block(up8, 64)
    conv8 = conv_block(conv8, 64)

    up9 = merge_block(conv8, conv1)
    conv9 = conv_block(up9, 48)
    conv9 = conv_block(conv9, 48)

    conv10 = Convolution2D(n_classes, 3, 3, activation='sigmoid', border_mode='same')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(2e-3), loss=combined_loss, metrics=[jaccard_coef, jaccard_coef_int])

    return model


def unet3(input_shapes, n_classes):
    inputs = Input(input_shapes['in'], name='in')

    conv1 = conv_block(inputs, 48)
    conv1 = conv_block(conv1, 64)
    pool1 = pool_block(conv1, 2)

    conv2 = conv_block(pool1, 64)
    conv2 = conv_block(conv2, 96)
    pool2 = pool_block(conv2, 2)

    conv3 = conv_block(pool2, 96)
    conv3 = conv_block(conv3, 128)
    pool3 = pool_block(conv3, 2)

    conv4 = conv_block(pool3, 128)
    conv4 = conv_block(conv4, 256)
    pool4 = pool_block(conv4, 2)

    conv5 = conv_block(pool4, 512)
    conv5 = conv_block(conv5, 256)

    up6 = merge_block(conv5, conv4, mode='sum')
    conv6 = conv_block(up6, 128)
    conv6 = conv_block(conv6, 128)

    up7 = merge_block(conv6, conv3, mode='sum')
    conv7 = conv_block(up7, 96)
    conv7 = conv_block(conv7, 96)

    up8 = merge_block(conv7, conv2, mode='sum')
    conv8 = conv_block(up8, 64)
    conv8 = conv_block(conv8, 64)

    up9 = merge_block(conv8, conv1, mode='sum')
    conv9 = conv_block(up9, 48)
    conv9 = conv_block(conv9, 48)

    conv10 = Convolution2D(n_classes, 3, 3, activation='sigmoid', border_mode='same')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int])

    return model


def rnet1(input_shapes, n_classes):
    def conv(size, x):
        x = Convolution2D(size, 3, 3, border_mode='same', init='he_normal', bias=False)(x)
        x = BatchNormalization(axis=1, mode=0)(x)
        x = PReLU(shared_axes=[2, 3])(x)
        return x

    def unet_block(sizes, inp):
        x = inp

        skips = []

        for sz in sizes[:-1]:
            x = conv(sz, x)
            skips.append(x)
            x = MaxPooling2D((2, 2))(x)

        x = conv(sizes[-1], x)

        for sz in reversed(sizes[:-1]):
            x = conv(sz, merge([UpSampling2D((2, 2))(x), skips.pop()], mode='concat', concat_axis=1))

        return x

    def fcn_block(sizes, inp):
        x = inp

        for sz in sizes:
            x = conv(sz, x)

        return Dropout(0.2)(x)

    # Build piramid of inputs
    inp0 = Input(input_shapes['in'], name='in')
    inp1 = AveragePooling2D((2, 2))(inp0)
    inp2 = AveragePooling2D((2, 2))(inp1)

    # Build outputs in resnet fashion
    out2 = unet_block([32, 48], inp2)
    #out2 = merge([unet_block([32, 48, 32], merge([inp2, out2], mode='concat', concat_axis=1)), out2], mode='sum')

    out1 = UpSampling2D((2, 2))(out2)
    #out1 = merge([unet_block([32, 32, 48], merge([inp1, out1], mode='concat', concat_axis=1)), out1], mode='sum')
    out1 = merge([unet_block([32, 48, 64], merge([inp1, out1], mode='concat', concat_axis=1)), out1], mode='sum')

    out0 = UpSampling2D((2, 2))(out1)
    out0 = merge([unet_block([32, 48, 64], merge([inp0, out0], mode='concat', concat_axis=1)), out0], mode='sum')
    out0 = merge([unet_block([32, 48, 64], merge([inp0, out0], mode='concat', concat_axis=1)), out0], mode='sum')

    # Final convolution
    out = Convolution2D(n_classes, 1, 1, activation='sigmoid')(out0)

    model = Model(input=inp0, output=out)
    model.compile(optimizer=Adam(2e-3), loss=combined_loss, metrics=[jaccard_coef, jaccard_coef_int])

    return model


def rnet1_mi(input_shapes, n_classes):
    def conv(size, x):
        x = Convolution2D(size, 3, 3, border_mode='same', init='he_normal', bias=False)(x)
        x = BatchNormalization(axis=1, mode=0)(x)
        x = PReLU(shared_axes=[2, 3])(x)
        return x

    def unet_block(sizes, inp):
        x = inp

        skips = []

        for sz in sizes[:-1]:
            x = conv(sz, x)
            skips.append(x)
            x = MaxPooling2D((2, 2))(x)

        x = conv(sizes[-1], x)

        for sz in reversed(sizes[:-1]):
            x = conv(sz, merge([UpSampling2D((2, 2))(x), skips.pop()], mode='concat', concat_axis=1))

        return x

    def radd(out, inp, block):
        block_in = merge([inp, out], mode='concat', concat_axis=1)
        block_out = block(block_in)

        return merge([block_out, out], mode='sum')

    in_I = Input(input_shapes['in_I'], name='in_I')
    in_M = Input(input_shapes['in_M'], name='in_M')

    # Build piramid of inputs
    inp0 = in_I
    inp1 = AveragePooling2D((2, 2))(inp0)
    inp2 = merge([AveragePooling2D((2, 2))(inp1), in_M], mode='concat', concat_axis=1)
    inp3 = AveragePooling2D((2, 2))(inp2)

    # Build outputs in resnet fashion
    out3 = unet_block([32, 48], inp3)

    out2 = UpSampling2D((2, 2))(out3)
    out2 = radd(out2, inp2, lambda x: unet_block([32, 48], x))

    out1 = UpSampling2D((2, 2))(out2)
    out1 = radd(out1, inp1, lambda x: unet_block([32, 48], x))
    out1 = radd(out1, inp1, lambda x: unet_block([32, 48, 64], x))

    out0 = UpSampling2D((2, 2))(out1)
    out0 = radd(out0, inp0, lambda x: unet_block([32, 48], x))
    out0 = radd(out0, inp0, lambda x: unet_block([32, 48, 64], x))

    # Final convolution
    out = Convolution2D(n_classes, 1, 1, activation='sigmoid')(out0)

    model = Model(input=[in_I, in_M], output=out)
    model.compile(optimizer=Adam(2e-3), loss=combined_loss, metrics=[jaccard_coef, jaccard_coef_int])

    return model


def unet_mi(input_shapes, n_classes):
    in_I = Input(input_shapes['in_I'], name='in_I')
    in_M = Input(input_shapes['in_M'], name='in_M')

    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(in_I)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(merge([pool2, in_M], mode='concat', concat_axis=1))
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

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

    model = Model(input=[in_I, in_M], output=conv10)
    model.compile(optimizer=Adam(), loss=combined_loss, metrics=[jaccard_coef, jaccard_coef_int])

    return model


def unet_mi_2(input_shapes, n_classes):
    in_I = Input(input_shapes['in_I'], name='in_I')
    in_M = Input(input_shapes['in_M'], name='in_M')

    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(in_I)
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(merge([pool2, in_M], mode='concat', concat_axis=1))
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(n_classes, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=[in_I, in_M], output=conv10)
    model.compile(optimizer=Adam(3e-3), loss=combined_loss, metrics=[jaccard_coef, jaccard_coef_int])

    return model


def unet3_mi(input_shapes, n_classes):
    in_I = Input(input_shapes['in_I'], name='in_I')
    in_M = Input(input_shapes['in_M'], name='in_M')

    conv1 = conv_block(in_I, 48)
    conv1 = conv_block(conv1, 64)
    pool1 = pool_block(conv1, 2)

    conv2 = conv_block(pool1, 64)
    conv2 = conv_block(conv2, 96)
    pool2 = pool_block(conv2, 2)

    conv3 = conv_block(merge([pool2, in_M], mode='concat', concat_axis=1), 96)
    conv3 = conv_block(conv3, 128)
    pool3 = pool_block(conv3, 2)

    conv4 = conv_block(pool3, 128)
    conv4 = conv_block(conv4, 256)
    pool4 = pool_block(conv4, 2)

    conv5 = conv_block(pool4, 256)
    conv5 = conv_block(conv5, 256)

    up6 = merge_block(conv5, conv4, mode='sum')
    conv6 = conv_block(up6, 128)
    conv6 = conv_block(conv6, 128)

    up7 = merge_block(conv6, conv3, mode='sum')
    conv7 = conv_block(up7, 96)
    conv7 = conv_block(conv7, 96)

    up8 = merge_block(conv7, conv2, mode='sum')
    conv8 = conv_block(up8, 64)
    conv8 = conv_block(conv8, 64)

    up9 = merge_block(conv8, conv1, mode='sum')
    conv9 = conv_block(up9, 48)
    conv9 = conv_block(conv9, 48)

    conv10 = Convolution2D(n_classes, 3, 3, activation='sigmoid', border_mode='same')(conv9)

    model = Model(input=[in_I, in_M], output=conv10)
    model.compile(optimizer=Adam(), loss=combined_loss, metrics=[jaccard_coef, jaccard_coef_int])

    return model
