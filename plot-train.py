import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shapely.wkt
import argparse

from util.data import grid_sizes, train_wkt
from util.masks import convert_geo_coords_to_raster, poly_to_mask

from matplotlib.patches import Polygon

cls_colors = {
    0: 'red',
    1: 'purple',
    2: 'black',
    3: 'gray',
    4: 'green',
    5: 'yellow',
    6: 'blue',
    7: 'cyan',
    8: 'white',
    9: 'yellow',
}

cls_alphas = {
    0: 0.3,
    1: 0.3,
    2: 0.5,
    3: 0.5,
    4: 0.5,
    5: 0.15,
    6: 0.15,
    7: 0.15,
    8: 0.4,
    9: 0.4,
}


def load_image(image_id):
    img = np.load('cache/images/%s_I.npy' % image_id).astype(np.float32)

    for c in xrange(img.shape[0]):
        l, h = np.percentile(img[c], [1, 99])

        img[c] = np.clip((img[c] - l) / (h - l), 0, 1)

    return img


def plot_overlay(image_id):
    img = load_image(image_id)

    fig, ax = plt.subplots()

    xmax = grid_sizes.loc[image_id, 'xmax']
    ymin = grid_sizes.loc[image_id, 'ymin']

    ax.imshow(np.rollaxis(img, 0, 3))

    # plotting, color by class type
    for cls in xrange(10):
        multi_poly = shapely.wkt.loads(train_wkt.loc[(train_wkt['cls'] == cls + 1) & (train_wkt['image_id'] == image_id), 'multi_poly_wkt'].values[0])

        for poly in multi_poly:
            coords = convert_geo_coords_to_raster(np.array(poly.exterior), img.shape[1:], (xmax, ymin))
            ax.add_patch(Polygon(coords, color=cls_colors[cls], lw=1.0, alpha=cls_alphas[cls]))

    plt.title(image_id)
    plt.show()


parser = argparse.ArgumentParser(description='Plot submission')
parser.add_argument('image', type=str, help='help image to plot')

args = parser.parse_args()

plot_overlay(args.image)
