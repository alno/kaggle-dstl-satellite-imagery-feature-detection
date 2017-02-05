import numpy as np

from util.meta import n_classes, class_names
from util.masks import mask_to_poly
from util.data import grid_sizes, train_wkt

import shapely.wkt

import sys


cls_opts = {
    8: {'min_area': 0.1, 'threshold': 0.1},
    9: {'min_area': 0.1, 'threshold': 0.1},
}

cls_thr = {
    1: 0.2,
    8: 0.3,
    9: 0.05
}


def pixel_jaccard(true_mask, pred_mask):
    return float((true_mask & pred_mask).sum()) / (true_mask | pred_mask).sum()


def poly_jaccard(true_poly, pred_poly):
    return pred_poly.intersection(true_poly).area / pred_poly.union(true_poly).area


def analyze_prediction(image_id, pred_id):
    xymax = (grid_sizes.loc[image_id, 'xmax'], grid_sizes.loc[image_id, 'ymin'])

    mask = np.load('cache/masks/%s.npy' % image_id)
    pred = np.load('cache/preds/%s.npy' % pred_id)

    pixel_jacs = np.zeros(n_classes)
    poly_jacs = np.zeros(n_classes)

    for cls in xrange(n_classes):
        thr = cls_thr.get(cls, 0.5)

        mask_poly = shapely.wkt.loads(train_wkt.loc[(train_wkt['image_id'] == image_id) & (train_wkt['cls'] == cls+1), 'multi_poly_wkt'].iloc[0])
        pred_poly = mask_to_poly(pred[cls] > thr, xymax, **cls_opts.get(cls, {}))

        pixel_jacs[cls] = pixel_jaccard(mask[cls], pred[cls] > thr)
        poly_jacs[cls] = poly_jaccard(mask_poly, pred_poly)

        print "Class %d (%s), thr=%.3f: pixel %.5f, poly %.5f" % (cls, class_names[cls], thr, pixel_jacs[cls], poly_jacs[cls])

    print "Total: pixel %.5f, poly %.5f" % (pixel_jacs.mean(), poly_jacs.mean())


analyze_prediction(sys.argv[1], sys.argv[2])
