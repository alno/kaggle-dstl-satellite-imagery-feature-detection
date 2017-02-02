from util.meta import locations
from util import save_pickle

import tifffile as tiff
import numpy as np
import cv2


def normalize(x):
    x = x.astype(np.float32)

    for c in xrange(x.shape[0]):
        qb, qt = np.percentile(x[c], [5, 95])

        x[c] = np.clip((x[c] - qb) / (qt - qb), 0, 1)

    return x


def read_location_images(loc, directory, kind=None):
    if kind is not None:
        suffix = '_' + kind
    else:
        suffix = ''

    imgs = []

    # Prepare images
    for i in xrange(5):
        imgs.append([])

        for j in xrange(5):
            img = normalize(tiff.imread('../input/%s/%s_%d_%d%s.tif' % (directory, loc, i, j, suffix)))
            imgs[i].append(img)

    return imgs


def write_location_images(loc, imgs, kind):
    ys = [0]
    xs = [0]

    for i in xrange(5):
        assert len(set(imgs[i][j].shape[1] for j in xrange(5))) == 1
        ys.append(ys[i] + imgs[i][0].shape[1])

    for i in xrange(5):
        assert len(set(imgs[j][i].shape[2] for j in xrange(5))) == 1
        xs.append(xs[i] + imgs[0][i].shape[2])

    data = np.zeros((imgs[0][0].shape[0], ys[-1], xs[-1]), dtype=np.float16)

    # Copy image data to the matrix
    for i in xrange(5):
        for j in xrange(5):
            img = imgs[i][j]
            ysz = img.shape[1]
            xsz = img.shape[2]

            data[:, ys[i]:ys[i]+ysz, xs[j]:xs[j]+xsz] = img.astype(np.float16)

    # Save images
    for i in xrange(5):
        for j in xrange(5):
            img = imgs[i][j]
            ysz = img.shape[1]
            xsz = img.shape[2]

            np.save('cache/images/%s_%d_%d_%s.npy' % (loc, i, j, kind), data[:, ys[i]:ys[i]+ysz, xs[j]:xs[j]+xsz])

    # Save debug location map
    if False:
        cv2.imwrite("%s.png" % loc, np.rollaxis(data * 255.0, 0, 3).astype(np.uint8))


print "Preparing image data..."

# Prepare location
for loc in locations:
    print "  Processing %s..." % loc

    imgs_i = read_location_images(loc, 'three_band')
    imgs_m = read_location_images(loc, 'sixteen_band', 'M')

    # Prepare images
    for i in xrange(5):
        for j in xrange(5):
            save_pickle('cache/meta/%s_%d_%d.pickle' % (loc, i, j), {'shape': imgs_i[i][j].shape, 'shape_m': imgs_m[i][j].shape})

    write_location_images(loc, imgs_i, 'I')
    write_location_images(loc, imgs_m, 'M')

print "Done."
