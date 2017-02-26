from util.meta import n_classes
from util.data import train_wkt, grid_sizes
from util.masks import poly_to_mask
from util import load_pickle

import numpy as np

import shapely.wkt as wkt

import cv2


print "Preparing train image masks..."

# Prepare location
for image_id, image_cls_wkt in train_wkt.groupby('image_id'):
    print "  Processing %s..." % image_id

    xmax = grid_sizes.loc[image_id, 'xmax']
    ymin = grid_sizes.loc[image_id, 'ymin']

    meta = load_pickle('cache/meta/%s.pickle' % image_id)

    mask = np.zeros((n_classes, meta['shape'][1], meta['shape'][2]), dtype=np.float32)

    for tp in image_cls_wkt.itertuples():
        poly = wkt.loads(tp.multi_poly_wkt)
        h, w = meta['shape'][1:]

        mask[tp.cls-1] = np.clip(cv2.resize(poly_to_mask(poly, (w * 4, h * 4), [xmax, ymin]).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA), 0, 1)

    np.save('cache/masks/%s.npy' % image_id, mask)

print "Done."
