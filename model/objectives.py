
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


def jaccard_coef(y_true, y_pred, smooth=1e-12, class_weights=1):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    union = K.sum(y_true + y_pred, axis=[0, -1, -2]) - intersection

    return K.mean((intersection + smooth) / (union + smooth) * class_weights)


def jaccard_coef_int(y_true, y_pred, smooth=1e-12, class_weights=1):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    union = K.sum(y_true + y_pred, axis=[0, -1, -2]) - intersection

    return K.mean((intersection + smooth) / (union + smooth) * class_weights)


def jaccard_coef_loss(y_true, y_pred, smooth=1e-3, class_priors=None, class_weights=1):
    inter = K.sum(y_true * y_pred, axis=[0, -1, -2])
    union = K.sum(K.square(y_true) + K.square(y_pred), axis=[0, -1, -2]) - inter

    if class_priors is None:
        inter += smooth
        union += smooth
    else:
        inter += smooth * class_priors
        union += smooth

    return 1.0 - K.mean((inter / union) * class_weights)


def combined_loss(y_true, y_pred, smooth=1e-3, class_priors=None, jac_weight=0.1, class_weights=1.0):
    cse = K.mean(K.mean(K.binary_crossentropy(y_pred, y_true), axis=[0, -1, -2]) * class_weights)
    jac = jaccard_coef_loss(y_true, y_pred, smooth, class_priors, class_weights)
    return (1-jac_weight) * cse + jac_weight * jac
