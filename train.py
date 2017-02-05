import numpy as np

import datetime
import time
import sys

import shapely.wkt

import argparse

from util.meta import val_train_image_ids, val_test_image_ids, full_train_image_ids
from util.data import grid_sizes, sample_submission
from util.masks import mask_to_poly

from model import ModelPipeline
from model.presets import presets

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('preset', type=str, help='model preset (features and hyperparams)')
parser.add_argument('--no-train', action='store_true', help='skip model training, just load current weights and predict')
parser.add_argument('--no-predict', action='store_true', help='skip model prediction')
parser.add_argument('--no-val', action='store_true', help='skip validation pass')
parser.add_argument('--no-full', action='store_true', help='skip full pass')
parser.add_argument('--cont', action='store_true', help='load prev weights and continue optimization')


args = parser.parse_args()

preset_name = args.preset
preset = presets[preset_name]

print "Using preset: %s" % preset_name

# Validation pass
if not args.no_val:
    print "Validation pass..."

    pipeline = ModelPipeline('%s-val' % preset_name, **preset)

    if args.no_train or args.cont:
        pipeline.load()

    if not args.no_train:
        pipeline.fit(val_train_image_ids, val_test_image_ids)

    if not args.no_predict:
        for image_id in val_test_image_ids:
            sys.stdout.write("  Processing %s... " % (image_id))
            sys.stdout.flush()

            start_time = time.time()

            p = pipeline.predict(image_id)
            np.save('cache/preds/%s-%s.npy' % (image_id, preset_name), p)

            print "Done in %d seconds" % (time.time() - start_time)


# Full pass
if not args.no_full:
    print "Full pass..."

    pipeline = ModelPipeline('%s-full' % preset_name, **preset)

    if args.no_train or args.cont:
        pipeline.load()

    if not args.no_train:
        pipeline.fit(full_train_image_ids)

    if not args.no_predict:
        subm = sample_submission.copy()

        for image_id in subm['ImageId'].unique():
            start_time = time.time()

            sys.stdout.write("  Processing %s... " % image_id)
            sys.stdout.flush()

            mask = pipeline.predict(image_id)
            xymax = (grid_sizes.loc[image_id, 'xmax'], grid_sizes.loc[image_id, 'ymin'])

            for cls in subm.loc[subm['ImageId'] == image_id, 'ClassType'].unique():
                subm.loc[(subm['ImageId'] == image_id) & (subm['ClassType'] == cls), 'MultipolygonWKT'] = shapely.wkt.dumps(mask_to_poly(mask[cls - 1], xymax))

            print "Done in %d seconds" % (time.time() - start_time)

        sys.stdout.write("Saving... ")
        sys.stdout.flush()

        subm_name = 'subm-%s-%s' % (preset_name, datetime.datetime.now().strftime('%Y%m%d-%H%M'))
        subm.to_csv('subm/%s.csv.gz' % subm_name, index=False, compression='gzip')

        print "Done, submission name: %s" % subm_name


print "Done."
