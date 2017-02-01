import numpy as np

import datetime
import shapely.wkt

from util.masks import mask_to_poly
from util.data import grid_sizes, sample_submission

subm = sample_submission.copy()

for i in subm.index:
    image_id = subm.loc[i, 'ImageId']
    xymax = (grid_sizes.loc[image_id, 'xmax'], grid_sizes.loc[image_id, 'ymin'])
    cls = subm.loc[i, 'ClassType']

    print "  Processing %s / %d..." % (image_id, cls)

    mask = np.load('cache/preds/%s.npy' % image_id)

    subm.loc[i, 'MultipolygonWKT'] = shapely.wkt.dumps(mask_to_poly(mask[cls - 1], xymax))

print "Saving..."

subm.to_csv('subm/subm-%s.csv.gz' % datetime.datetime.now().strftime('%Y%m%d-%H%M'), index=False, compression='gzip')

print "Done."
