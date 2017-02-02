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
from model.archs import unet, unet2

presets = {
    'u1d3': {
        'arch': unet,
        'n_epoch': 100,
        'patch_size': 64,
        'downscale': 3,
    },

    'u1d2': {
        'arch': unet,
        'n_epoch': 100,
        'patch_size': 64,
        'downscale': 2,
    },

    'u2d3': {
        'arch': unet2,
        'n_epoch': 100,
        'patch_size': 64,
        'downscale': 3,
    },

    'u1d3b': {
        'arch': unet,
        'n_epoch': 100,
        'patch_size': 64,
        'downscale': 3,
        'classes': [0, 4, 5, 6]
    },

    'u1d3s': {
        'arch': unet,
        'n_epoch': 100,
        'patch_size': 64,
        'downscale': 1,
        'classes': [7, 8, 9],
        'batch_mode': 'random',
    },

    'u1d3br': {
        'arch': unet,
        'n_epoch': 100,
        'patch_size': 64,
        'downscale': 3,
        'classes': [0, 4, 5, 6],
        'batch_mode': 'random',
    },

    'u1d4br': {
        'arch': unet,
        'n_epoch': 100,
        'patch_size': 64,
        'downscale': 4,
        'classes': [4, 5, 6],
        'batch_mode': 'random',
    },
}

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('preset', type=str, help='model preset (features and hyperparams)')
parser.add_argument('--no-train', action='store_true', help='skip model training, just load current weights and predict')
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

    for image_id in val_test_image_ids:
        p = pipeline.predict(image_id)
        np.save('cache/preds/%s.npy' % image_id, p)


# Full pass
if not args.no_full:
    print "Full pass..."

    pipeline = ModelPipeline('%s-full' % preset_name, **preset)

    if args.no_train or args.cont:
        pipeline.load()

    if not args.no_train:
        pipeline.fit(full_train_image_ids)

    subm = sample_submission.copy()

    for i in subm.index:
        image_id = subm.loc[i, 'ImageId']
        xymax = (grid_sizes.loc[image_id, 'xmax'], grid_sizes.loc[image_id, 'ymin'])
        cls = subm.loc[i, 'ClassType']

        sys.stdout.write("  Processing %s / %d... " % (image_id, cls))
        sys.stdout.flush()

        start_time = time.time()
        mask = pipeline.predict(image_id)

        subm.loc[i, 'MultipolygonWKT'] = shapely.wkt.dumps(mask_to_poly(mask[cls - 1], xymax))

        print "Done in %d seconds" % (time.time() - start_time)

    sys.stdout.write("Saving... ")
    sys.stdout.flush()

    subm_name = 'subm-%s-%s' % (preset_name, datetime.datetime.now().strftime('%Y%m%d-%H%M'))
    subm.to_csv('subm/%s.csv.gz' % subm_name, index=False, compression='gzip')

    print "Done, submission name: %s" % subm_name


print "Done."
