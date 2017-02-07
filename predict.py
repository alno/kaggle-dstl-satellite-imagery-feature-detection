import numpy as np

import datetime
import time
import sys

import shapely.wkt

from util.data import grid_sizes, sample_submission, train_wkt
from util.masks import mask_to_poly
from util.meta import n_classes, val_test_image_ids, class_names

from model import ModelPipeline
from model.presets import presets

cls_opts = {
}

cls_thr = {
    1: 0.2
}


def load_model(preset_name, split_name):
    pipeline = ModelPipeline('%s-%s' % (preset_name, split_name), **presets[preset_name])
    pipeline.load()
    return pipeline


def combine(masks):
    mask = masks['r1m'].copy()

    for i in [0, 1, 2, 3]:
        mask[i] = masks['u3mi_structs'][i]

    return mask


model_names = ['r1m', 'u3mi_structs']


if True:
    print "Validation pass, loading models..."

    models = dict(zip(model_names, [load_model(m, 'val') for m in model_names]))

    print "Validating..."

    pixel_intersections = np.zeros(n_classes)
    pixel_unions = np.zeros(n_classes) + 1e-12

    poly_intersections = np.zeros(n_classes)
    poly_unions = np.zeros(n_classes) + 1e-12

    for image_id in val_test_image_ids:
        start_time = time.time()

        sys.stdout.write("  Processing %s... " % image_id)
        sys.stdout.flush()

        mask = np.load('cache/masks/%s.npy' % image_id)
        pred = combine(dict(zip(model_names, [models[m].predict(image_id) for m in model_names])))
        xymax = (grid_sizes.loc[image_id, 'xmax'], grid_sizes.loc[image_id, 'ymin'])

        for cls in xrange(n_classes):
            cls_pred = pred[cls] > cls_thr.get(cls, 0.5)
            cls_pixel_inter = (cls_pred & mask[cls]).sum()
            cls_pixel_union = (cls_pred | mask[cls]).sum()

            pixel_intersections[cls] += cls_pixel_inter
            pixel_unions[cls] += cls_pixel_union

            true_poly = shapely.wkt.loads(train_wkt.loc[(train_wkt['image_id'] == image_id) & (train_wkt['cls'] == cls+1), 'multi_poly_wkt'].iloc[0])
            pred_poly = mask_to_poly(cls_pred, xymax, **cls_opts.get(cls, {}))

            poly_intersections[cls] += pred_poly.intersection(true_poly).area
            poly_unions[cls] += pred_poly.union(true_poly).area

        print "Done in %d seconds" % (time.time() - start_time)

    pixel_jacs = np.zeros(n_classes)
    poly_jacs = np.zeros(n_classes)
    for cls in xrange(n_classes):
        pixel_jacs[cls] = pixel_intersections[cls] / pixel_unions[cls]
        poly_jacs[cls] = poly_intersections[cls] / poly_unions[cls]

        print "Class %d (%s), thr=%.3f: pixel %.5f, poly %.5f" % (cls, class_names[cls], cls_thr.get(cls, 0.5), pixel_jacs[cls], poly_jacs[cls])

    print "Total: pixel %.5f, poly %.5f" % (pixel_jacs.mean(), poly_jacs.mean())

if True:
    print "Full pass, loading models..."

    models = dict(zip(model_names, [load_model(m, 'full') for m in model_names]))

    print "Predicting..."

    subm = sample_submission.copy()

    for image_id in subm['ImageId'].unique():
        start_time = time.time()

        sys.stdout.write("  Processing %s... " % image_id)
        sys.stdout.flush()

        pred = combine(dict(zip(model_names, [models[m].predict(image_id) for m in model_names])))
        xymax = (grid_sizes.loc[image_id, 'xmax'], grid_sizes.loc[image_id, 'ymin'])

        for cls in subm.loc[subm['ImageId'] == image_id, 'ClassType'].unique():
            cls_mask = pred[cls - 1] > cls_thr.get(cls-1, 0.5)
            cls_poly = mask_to_poly(cls_mask, xymax, **cls_opts.get(cls-1, {}))

            subm.loc[(subm['ImageId'] == image_id) & (subm['ClassType'] == cls), 'MultipolygonWKT'] = shapely.wkt.dumps(cls_poly, rounding_precision=8)

        print "Done in %d seconds" % (time.time() - start_time)

    sys.stdout.write("Saving... ")
    sys.stdout.flush()

    subm_name = 'subm-%s-%s' % ('multi', datetime.datetime.now().strftime('%Y%m%d-%H%M'))
    subm.to_csv('subm/%s.csv.gz' % subm_name, index=False, compression='gzip')

    print "Submission name: %s" % subm_name

print "Done."
