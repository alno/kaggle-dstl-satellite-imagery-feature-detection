import numpy as np

from util.meta import full_train_image_ids
from util.data import sample_submission, grid_sizes
from util.masks import mask_to_poly

from skimage.morphology import disk, binary_opening, binary_closing

import cv2
import time
import datetime
import sys
import shapely.wkt

import tifffile as tiff
import xgboost as xgb

cls = 7

pos_sample = 1
neg_sample = 0.1

X = []
y = []

print "Loading train images..."
for image_id in full_train_image_ids:
    print "  %s..." % image_id

    img = tiff.imread('../input/sixteen_band/%s_M.tif' % image_id)

    mask = np.load('cache/masks/%s.npy' % image_id)
    mask = cv2.resize(mask[cls], (img.shape[2], img.shape[1]), interpolation=cv2.INTER_AREA) > 0.5

    img_X = img.reshape((img.shape[0], img.shape[1] * img.shape[2])).T
    img_y = mask.reshape(img.shape[1] * img.shape[2])

    pos_idx = np.where(img_y >= 0.5)[0]
    neg_idx = np.where(img_y < 0.5)[0]

    if pos_sample < 1:
        pos_idx = np.random.choice(pos_idx, int(len(pos_idx) * pos_sample), replace=False)

    if neg_sample < 1:
        neg_idx = np.random.choice(neg_idx, int(len(neg_idx) * neg_sample), replace=False)

    idx = np.hstack((pos_idx, neg_idx))

    X.append(img_X[idx])
    y.append(img_y[idx])

X = np.vstack(X)
y = np.hstack(y)

dtrain = xgb.DMatrix(X, label=y)

params = {'objective': 'binary:logistic', 'silent': 1}

print "Validating..."
xgb.cv(params, dtrain, 200, verbose_eval=True)

print "Training..."
model = xgb.train(params, dtrain, 200, verbose_eval=True)

print "Predicting test..."
subm = sample_submission.copy()

for image_id in sorted(subm['ImageId'].unique()):
    start_time = time.time()

    sys.stdout.write("  Processing %s... " % image_id)
    sys.stdout.flush()

    img = tiff.imread('../input/sixteen_band/%s_M.tif' % image_id)

    mask = model.predict(xgb.DMatrix(img.reshape((img.shape[0], img.shape[1] * img.shape[2])).T)).reshape(img.shape[1:]) > 0.5
    mask = binary_closing(mask, disk(1))
    mask = binary_opening(mask, disk(1))
    xymax = (grid_sizes.loc[image_id, 'xmax'], grid_sizes.loc[image_id, 'ymin'])

    subm.loc[subm['ImageId'] == image_id, 'MultipolygonWKT'] = 'MULTIPOLYGON EMPTY'
    subm.loc[(subm['ImageId'] == image_id) & (subm['ClassType'] == cls + 1), 'MultipolygonWKT'] = shapely.wkt.dumps(mask_to_poly(mask, xymax), rounding_precision=8)

    print "Done in %d seconds" % (time.time() - start_time)

sys.stdout.write("Saving... ")
sys.stdout.flush()

subm_name = 'subm-xgb-%d-%s' % (cls, datetime.datetime.now().strftime('%Y%m%d-%H%M'))
subm.to_csv('subm/%s.csv.gz' % subm_name, index=False, compression='gzip')

print "Done, submission name: %s" % subm_name
