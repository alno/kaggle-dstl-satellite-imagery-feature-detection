import numpy as np
import matplotlib.pyplot as plt

from util.masks import poly_to_mask, mask_to_poly
from util.data import grid_sizes


def plot_prediction(image_id, cls=0):
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

    plt.show()


plot_prediction('6100_2_2')
