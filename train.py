import numpy as np

import datetime
import time
import sys

import shapely.wkt

import argparse

from util.meta import val_train_image_ids, val_test_image_ids, full_train_image_ids, n_classes, class_names
from util.data import grid_sizes, sample_submission, train_wkt
from util.masks import mask_to_poly

from model import ModelPipeline
from model.presets import presets


cls_opts = {
    0: {'epsilon': 1.0},
    1: {'epsilon': 0.1, 'min_area': 0.1},
}

cls_thr = {
    1: 0.3
}


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('preset', type=str, help='model preset (features and hyperparams)')
parser.add_argument('--no-train', action='store_true', help='skip model training, just load current weights and predict')
parser.add_argument('--no-predict', action='store_true', help='skip model prediction')
parser.add_argument('--no-val', action='store_true', help='skip validation pass')
parser.add_argument('--no-full', action='store_true', help='skip full pass')
parser.add_argument('--cont', type=int, help='load prev weights and continue optimization from given train stage')


args = parser.parse_args()

preset_name = args.preset
preset = presets[preset_name]

preset_opts = dict((k, v) for k, v in preset.items() if k != 'train')
preset_train_stages = preset['train'][args.cont or 0:]

print "Using preset: %s" % preset_name

# Validation pass
if not args.no_val:
    print "Validation pass..."

    pipeline = ModelPipeline('%s-val' % preset_name, **preset_opts)

    if args.no_train or args.cont:
        pipeline.load()

    if not args.no_train:
        for train_preset in preset_train_stages:
            print "Fitting with %s..." % str(train_preset)

            pipeline.fit(val_train_image_ids, val_test_image_ids, **train_preset)

    if not args.no_predict:

        pixel_intersections = np.zeros(n_classes)
        pixel_unions = np.zeros(n_classes) + 1e-12

        poly_intersections = np.zeros(n_classes)
        poly_unions = np.zeros(n_classes) + 1e-12

        for image_id in val_test_image_ids:
            sys.stdout.write("  Processing %s... " % (image_id))
            sys.stdout.flush()

            start_time = time.time()

            mask = np.load('cache/masks/%s.npy' % image_id)
            pred = pipeline.predict(image_id)
            np.save('cache/preds/%s-%s.npy' % (image_id, preset_name), pred)

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

# Full pass
if not args.no_full:
    print "Full pass..."

    pipeline = ModelPipeline('%s-full' % preset_name, **preset_opts)

    if args.no_train or args.cont:
        pipeline.load()
    else:
        pipeline.load_weights('%s-val' % preset_name)

    if not args.no_train:
        for train_preset in preset_train_stages:
            if preset.get('batch_mode') == 'random':
                train_preset = train_preset.copy()
                train_preset['n_epoch'] = int(train_preset['n_epoch'] * len(full_train_image_ids) / len(val_train_image_ids))

            print "Fitting with %s..." % str(train_preset)

            pipeline.fit(val_train_image_ids, val_test_image_ids, **train_preset)

    if not args.no_predict:
        subm = sample_submission.copy()

        for image_id in sorted(subm['ImageId'].unique()):
            start_time = time.time()

            sys.stdout.write("  Processing %s... " % image_id)
            sys.stdout.flush()

            mask = pipeline.predict(image_id)
            xymax = (grid_sizes.loc[image_id, 'xmax'], grid_sizes.loc[image_id, 'ymin'])

            for cls in subm.loc[subm['ImageId'] == image_id, 'ClassType'].unique():
                subm.loc[(subm['ImageId'] == image_id) & (subm['ClassType'] == cls), 'MultipolygonWKT'] = shapely.wkt.dumps(mask_to_poly(mask[cls - 1], xymax), rounding_precision=8)

            print "Done in %d seconds" % (time.time() - start_time)

        sys.stdout.write("Saving... ")
        sys.stdout.flush()

        subm_name = 'subm-%s-%s' % (preset_name, datetime.datetime.now().strftime('%Y%m%d-%H%M'))
        subm.to_csv('subm/%s.csv.gz' % subm_name, index=False, compression='gzip')

        print "Done, submission name: %s" % subm_name


print "Done."
