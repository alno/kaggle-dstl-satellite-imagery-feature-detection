import numpy as np
import matplotlib.pyplot as plt

import sys

from util.masks import poly_to_mask, mask_to_poly
from util.data import grid_sizes


def plot_prediction(image_id, pred_id, cls=0):
    image = np.load('cache/images/%s.npy' % image_id)
    pred = np.load('cache/preds/%s.npy' % image_id)

    xymax = (grid_sizes.loc[image_id, 'xmax'], grid_sizes.loc[image_id, 'ymin'])

    plt.figure()

    ax1 = plt.subplot(131)
    ax1.set_title('image_id:%s' % image_id)
    ax1.imshow(image[1, :, :], cmap=plt.get_cmap('gray'))
    ax2 = plt.subplot(132)
    ax2.set_title('predict bldg pixels')
    ax2.imshow(pred[cls], cmap=plt.get_cmap('hot'))
    ax3 = plt.subplot(133)
    ax3.set_title('predict bldg polygones')
    ax3.imshow(poly_to_mask(mask_to_poly(pred[cls], xymax), image.shape[1:], xymax), cmap=plt.get_cmap('hot'))

    plt.title("%s - class %d" % (pred_id, cls))
    plt.show()


def plot_all_class_predictions(image_id, pred_id):
    image = np.load('cache/images/%s_I.npy' % image_id)
    mask = np.load('cache/masks/%s.npy' % image_id)
    pred = np.load('cache/preds/%s.npy' % pred_id)

    f, ax = plt.subplots(2, 5, sharex='col', sharey='row')

    ax[0][0].imshow(np.rollaxis(image.astype(np.float32) / image.max(axis=(1, 2))[:, np.newaxis, np.newaxis], 0, 3))

    for c in xrange(9):
        ax[(c+1) // 5][(c+1) % 5].imshow(np.dstack((mask[c], pred[c], np.zeros(mask[c].shape))))

    plt.title(pred_id)
    plt.tight_layout()
    plt.show()


def plot_class_prediction(image_id, pred_id, c):
    mask = np.load('cache/masks/%s.npy' % image_id)
    pred = np.load('cache/preds/%s.npy' % pred_id)

    plt.title("%s - class %d" % (pred_id, c))
    plt.imshow(np.dstack((mask[c], pred[c], np.zeros(mask[c].shape))))
    plt.show()


image_id = sys.argv[1][:8]
pred_id = sys.argv[1]

if len(sys.argv) > 2:
    plot_class_prediction(image_id, pred_id, int(sys.argv[2]))
else:
    plot_all_class_predictions(image_id, pred_id)
