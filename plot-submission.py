import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shapely.wkt
import argparse

from util.data import grid_sizes
from util.masks import convert_geo_coords_to_raster, poly_to_mask

from matplotlib.patches import Polygon

cls_colors = {
    0: 'red',
    1: 'red',
    2: 'gray',
    3: 'gray',
    4: 'green',
    5: 'yellow',
    6: 'blue',
    7: 'cyan',
    8: 'purple',
    9: 'purple',
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
    8: 0.3,
    9: 0.3,
}


def load_image(image_id):
    img = np.load('cache/images/%s_I.npy' % image_id).astype(np.float32)

    for c in xrange(img.shape[0]):
        l, h = np.percentile(img[c], [1, 99])

        img[c] = np.clip((img[c] - l) / (h - l), 0, 1)

    return img


def plot_prediction_overlay(subm_name, image_id):
    subm = pd.read_csv('subm/%s.csv.gz' % subm_name)
    img = load_image(image_id)

    fig, ax = plt.subplots()

    xmax = grid_sizes.loc[image_id, 'xmax']
    ymin = grid_sizes.loc[image_id, 'ymin']

    ax.imshow(np.rollaxis(img, 0, 3))

    # plotting, color by class type
    for cls in xrange(10):
        multi_poly = shapely.wkt.loads(subm.loc[(subm['ClassType'] == cls + 1) & (subm['ImageId'] == image_id), 'MultipolygonWKT'].values[0])

        for poly in multi_poly:
            coords = convert_geo_coords_to_raster(np.array(poly.exterior), img.shape[1:], (xmax, ymin))
            ax.add_patch(Polygon(coords, color=cls_colors[cls], lw=1.0, alpha=cls_alphas[cls]))

    plt.title(subm_name)
    plt.show()


def plot_predictions(subm_name, image_id):
    subm = pd.read_csv('subm/%s.csv.gz' % subm_name)
    image = load_image(image_id)

    xmax = grid_sizes.loc[image_id, 'xmax']
    ymin = grid_sizes.loc[image_id, 'ymin']

    f, ax = plt.subplots(2, 5, sharex='col', sharey='row')

    ax[0][0].imshow(np.rollaxis(image, 0, 3))

    for c in xrange(9):
        multi_poly = shapely.wkt.loads(subm.loc[(subm['ClassType'] == c + 1) & (subm['ImageId'] == image_id), 'MultipolygonWKT'].values[0])

        ax[(c+1) // 5][(c+1) % 5].imshow(poly_to_mask(multi_poly, image.shape[1:], [xmax, ymin]), cmap='hot')

    plt.tight_layout()
    plt.show()


def plot_class_prediction(subm_name, image_id, c):
    subm = pd.read_csv('subm/%s.csv.gz' % subm_name)
    image = load_image(image_id)

    xmax = grid_sizes.loc[image_id, 'xmax']
    ymin = grid_sizes.loc[image_id, 'ymin']

    multi_poly = shapely.wkt.loads(subm.loc[(subm['ClassType'] == c + 1) & (subm['ImageId'] == image_id), 'MultipolygonWKT'].values[0])

    plt.title("%s - class %d" % (subm_name, c))
    plt.imshow(poly_to_mask(multi_poly, image.shape[1:], [xmax, ymin]), cmap='hot')
    plt.show()


parser = argparse.ArgumentParser(description='Plot submission')
parser.add_argument('subm', type=str, help='submission name')
parser.add_argument('--overlay', action='store_true', help='plot class overlay')
parser.add_argument('--cls', type=int, help='class index')
parser.add_argument('--image', type=str, default='6100_0_2', help='help image to plot')

args = parser.parse_args()

if args.overlay:
    plot_prediction_overlay(args.subm, args.image)
elif args.cls is not None:
    plot_class_prediction(args.subm, args.image, args.cls)
else:
    plot_predictions(args.subm, args.image)
