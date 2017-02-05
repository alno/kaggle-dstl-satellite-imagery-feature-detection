from util.meta import locations
from util import save_pickle

import tifffile as tiff
import numpy as np
import cv2

n_location_images = 5


def read_location_images(loc, directory, kind=None):
    if kind is not None:
        suffix = '_' + kind
    else:
        suffix = ''

    imgs = []

    # Prepare images
    for i in xrange(n_location_images):
        imgs.append([])

        for j in xrange(n_location_images):
            img = tiff.imread('../input/%s/%s_%d_%d%s.tif' % (directory, loc, i, j, suffix))
            imgs[i].append(img)

    return imgs


def write_location_images(loc, imgs, kind):
    ys = [0]
    xs = [0]

    for i in xrange(n_location_images):
        assert len(set(imgs[i][j].shape[1] for j in xrange(n_location_images))) == 1
        ys.append(ys[i] + imgs[i][0].shape[1])

    for i in xrange(n_location_images):
        assert len(set(imgs[j][i].shape[2] for j in xrange(n_location_images))) == 1
        xs.append(xs[i] + imgs[0][i].shape[2])

    data = np.zeros((imgs[0][0].shape[0], ys[-1], xs[-1]), dtype=np.uint16)

    # Copy image data to the matrix
    for i in xrange(n_location_images):
        for j in xrange(n_location_images):
            img = imgs[i][j]
            ysz = img.shape[1]
            xsz = img.shape[2]

            data[:, ys[i]:ys[i]+ysz, xs[j]:xs[j]+xsz] = img

    # Save images
    for i in xrange(n_location_images):
        for j in xrange(n_location_images):
            img = imgs[i][j]
            ysz = img.shape[1]
            xsz = img.shape[2]

            np.save('cache/images/%s_%d_%d_%s.npy' % (loc, i, j, kind), data[:, ys[i]:ys[i]+ysz, xs[j]:xs[j]+xsz])

    # Save debug location map
    if False:
        cv2.imwrite("%s.png" % loc, np.rollaxis((data - data.min()) * 255.0 / (data.max() - data.min()), 0, 3).astype(np.uint8))


print "Preparing image data..."

# Prepare location
for loc in locations:
    print "  Processing %s..." % loc

    imgs_i = read_location_images(loc, 'three_band')
    imgs_m = read_location_images(loc, 'sixteen_band', 'M')

    # Prepare images
    for i in xrange(n_location_images):
        for j in xrange(n_location_images):
            save_pickle('cache/meta/%s_%d_%d.pickle' % (loc, i, j), {'shape': imgs_i[i][j].shape, 'shape_m': imgs_m[i][j].shape})

    write_location_images(loc, imgs_i, 'I')
    write_location_images(loc, imgs_m, 'M')

print "Done."
