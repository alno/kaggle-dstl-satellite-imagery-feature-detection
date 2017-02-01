from util.meta import mask_size, n_classes
from util.data import train_wkt, grid_sizes
from util.masks import poly_to_mask

import numpy as np

import shapely.wkt as wkt


print "Preparing train image masks..."

# Prepare location
for image_id, image_cls_wkt in train_wkt.groupby('image_id'):
    print "  Processing %s..." % image_id

    xmax = grid_sizes.loc[image_id, 'xmax']
    ymin = grid_sizes.loc[image_id, 'ymin']

    mask = np.zeros((n_classes, mask_size, mask_size), dtype=np.uint8)

    for tp in image_cls_wkt.itertuples():
        poly = wkt.loads(tp.multi_poly_wkt)
        mask[tp.cls-1] = poly_to_mask(poly, (mask_size, mask_size), [xmax, ymin])

    np.save('cache/masks/%s.npy' % image_id, mask)

print "Done."
