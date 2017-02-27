import numpy as np

import datetime
import time
import sys

import shapely.wkt

from util.data import grid_sizes, sample_submission, train_wkt
from util.masks import mask_to_poly, poly_to_mask
from util.meta import n_classes, full_train_image_ids, class_names


min_water_area = 0.03


def predict_mask(image_id):
    m = np.load('cache/images/%s_M.npy' % image_id).astype(np.float64)

    #r = m[4]
    re = m[5]
    mir = m[7]

    ccci = - (mir - re) / (mir + re)

    msk = (ccci > 0.11).astype(np.uint8)

    if msk.sum() < min_water_area * np.product(msk.shape):
        msk[:, :] = 0

    return msk[np.newaxis, :, :]


cls_opts = {'min_area': 100, 'threshold': 0.5}
classes = [6]


if True:
    print "Validating..."

    pixel_intersections = np.zeros(n_classes)
    pixel_unions = np.zeros(n_classes) + 1e-12

    poly_intersections = np.zeros(n_classes)
    poly_unions = np.zeros(n_classes) + 1e-12

    for image_id in full_train_image_ids:
        start_time = time.time()

        sys.stdout.write("  Processing %s... " % image_id)
        sys.stdout.flush()

        pred = predict_mask(image_id)
        xymax = (grid_sizes.loc[image_id, 'xmax'], grid_sizes.loc[image_id, 'ymin'])

        for ci, cls in enumerate(classes):
            true_poly = shapely.wkt.loads(train_wkt.loc[(train_wkt['image_id'] == image_id) & (train_wkt['cls'] == cls+1), 'multi_poly_wkt'].iloc[0])
            true_mask = poly_to_mask(true_poly, pred.shape[1:], xymax)

            pred_mask = pred[ci]

            cls_pixel_inter = (pred_mask & true_mask).sum()
            cls_pixel_union = (pred_mask | true_mask).sum()

            pixel_intersections[cls] += cls_pixel_inter
            pixel_unions[cls] += cls_pixel_union

            pred_poly = mask_to_poly(pred_mask, xymax, **cls_opts)

            if pred_poly.area < min_water_area * abs(np.product(xymax)):
                pred_poly = mask_to_poly(pred_mask * 0, xymax, **cls_opts)

            cls_poly_inter = pred_poly.intersection(true_poly).area
            cls_poly_union = pred_poly.union(true_poly).area

            poly_intersections[cls] += cls_poly_inter
            poly_unions[cls] += cls_poly_union

            print "%s: %.8f vs %.8f" % (image_id, pred_poly.area, true_poly.area)

        print "Done in %d seconds" % (time.time() - start_time)

    pixel_jacs = np.zeros(n_classes)
    poly_jacs = np.zeros(n_classes)
    for cls in xrange(n_classes):
        pixel_jacs[cls] = pixel_intersections[cls] / pixel_unions[cls]
        poly_jacs[cls] = poly_intersections[cls] / poly_unions[cls]

        print "Class %d (%s): pixel %.5f, poly %.5f" % (cls, class_names[cls], pixel_jacs[cls], poly_jacs[cls])

    print "Total: pixel %.5f, poly %.5f" % (pixel_jacs.mean(), poly_jacs.mean())


if True:
    print "Predicting..."

    subm = sample_submission.copy()

    for image_id in sorted(subm['ImageId'].unique()):
        start_time = time.time()

        sys.stdout.write("  Processing %s... " % image_id)
        sys.stdout.flush()

        pred = predict_mask(image_id)
        xymax = (grid_sizes.loc[image_id, 'xmax'], grid_sizes.loc[image_id, 'ymin'])

        for cls in subm.loc[subm['ImageId'] == image_id, 'ClassType'].unique():
            if (cls - 1) in classes:
                ci = classes.index(cls-1)

                pred_mask = pred[ci]
                pred_poly = mask_to_poly(pred_mask, xymax, **cls_opts)

                if pred_poly.area < min_water_area * abs(np.product(xymax)):
                    pred_poly = mask_to_poly(pred_mask * 0, xymax, **cls_opts)

                subm.loc[(subm['ImageId'] == image_id) & (subm['ClassType'] == cls), 'MultipolygonWKT'] = shapely.wkt.dumps(pred_poly, rounding_precision=8)
            else:
                subm.loc[(subm['ImageId'] == image_id) & (subm['ClassType'] == cls), 'MultipolygonWKT'] = 'MULTIPOLYGON EMPTY'

        print "Done in %d seconds" % (time.time() - start_time)

    sys.stdout.write("Saving... ")
    sys.stdout.flush()

    subm_name = 'subm-%s-%s' % ('water2', datetime.datetime.now().strftime('%Y%m%d-%H%M'))
    subm.to_csv('subm/%s.csv.gz' % subm_name, index=False, compression='gzip')

    print "Submission name: %s" % subm_name

print "Done."
