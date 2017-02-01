from util.meta import locations
from util import save_pickle

import tifffile as tiff
import numpy as np
import cv2


def normalize(x):
    x = x.astype(np.float16)

    for c in xrange(x.shape[0]):
        qb, qt = np.percentile(x[c], [5, 95])

        x[c] = np.clip((x[c] - qb) / (qt - qb), 0, 1)

    return x


print "Preparing image data..."

# Prepare location
for loc in locations:
    print "  Processing %s..." % loc

    imgs = []

    # Prepare images
    for i in xrange(5):
        imgs.append([])

        for j in xrange(5):
            img = normalize(tiff.imread('../input/three_band/%s_%d_%d.tif' % (loc, i, j)))
            imgs[i].append(img)

            save_pickle('cache/meta/%s_%d_%d.pickle' % (loc, i, j), {'shape': img.shape})

    # Build arrays of offsets
    ys = [0]
    xs = [0]

    for i in xrange(5):
        assert len(set(imgs[i][j].shape[1] for j in xrange(5))) == 1
        ys.append(ys[i] + imgs[i][0].shape[1])

    for i in xrange(5):
        assert len(set(imgs[j][i].shape[2] for j in xrange(5))) == 1
        xs.append(xs[i] + imgs[0][i].shape[2])

    data = np.zeros((3, ys[-1], xs[-1]), dtype=np.float16)

    # Copy image data to the matrix
    for i in xrange(5):
        for j in xrange(5):
            img = imgs[i][j]
            ysz = img.shape[1]
            xsz = img.shape[2]

            for c in xrange(3):
                data[:, ys[i]:ys[i]+ysz, xs[j]:xs[j]+xsz] = img.astype(np.float16)

    # Save images
    for i in xrange(5):
        for j in xrange(5):
            img = imgs[i][j]
            ysz = img.shape[1]
            xsz = img.shape[2]

            np.save('cache/images/%s_%d_%d.npy' % (loc, i, j), data[:, ys[i]:ys[i]+ysz, xs[j]:xs[j]+xsz].astype(np.float16))

    # Save debug location map
    if False:
        cv2.imwrite("%s.png" % loc, np.rollaxis(data * 255.0, 0, 3).astype(np.uint8))

print "Done."
