import numpy as np

import datetime
import time
import sys

import shapely.wkt

from util.data import grid_sizes, sample_submission
from util.masks import mask_to_poly

from model import ModelPipeline
from model.presets import presets


def load_model(preset_name):
    pipeline = ModelPipeline('%s-full' % preset_name, **presets[preset_name])
    pipeline.load()
    return pipeline

print "Loading models..."

models = {}
models['umi_struct'] = load_model('umi_struct')
models['um_areas'] = load_model('um_areas')

print "Predicting..."

subm = sample_submission.copy()

for image_id in subm['ImageId'].unique():
    start_time = time.time()

    sys.stdout.write("  Processing %s... " % image_id)
    sys.stdout.flush()

    mask = models['umi_struct'].predict(image_id) + models['um_areas'].predict(image_id)
    xymax = (grid_sizes.loc[image_id, 'xmax'], grid_sizes.loc[image_id, 'ymin'])

    for cls in subm.loc[subm['ImageId'] == image_id, 'ClassType'].unique():
        subm.loc[(subm['ImageId'] == image_id) & (subm['ClassType'] == cls), 'MultipolygonWKT'] = shapely.wkt.dumps(mask_to_poly(mask[cls - 1], xymax))

    print "Done in %d seconds" % (time.time() - start_time)

sys.stdout.write("Saving... ")
sys.stdout.flush()

subm_name = 'subm-%s-%s' % ('multi', datetime.datetime.now().strftime('%Y%m%d-%H%M'))
subm.to_csv('subm/%s.csv.gz' % subm_name, index=False, compression='gzip')

print "Done, submission name: %s" % subm_name
