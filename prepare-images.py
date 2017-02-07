from util.meta import locations
from util import load_pickle, save_pickle

from skimage.filters import sobel

import tifffile as tiff
import numpy as np
import cv2

n_location_images = 5


def resize(src, shape):
    dst = np.empty(shape=(src.shape[0], shape[0], shape[1]))

    for c in xrange(src.shape[0]):
        dst[c] = cv2.resize(src[c], (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

    return dst


def read_location_images(loc, directory, band=None, resize_to=None):
    if band is not None:
        suffix = '_' + band
    else:
        suffix = ''

    imgs = []

    # Prepare images
    for i in xrange(n_location_images):
        imgs.append([])

        for j in xrange(n_location_images):
            img = tiff.imread('../input/%s/%s_%d_%d%s.tif' % (directory, loc, i, j, suffix))

            if resize_to is not None:
                meta = load_pickle('cache/meta/%s_%d_%d.pickle' % (loc, i, j))
                img = resize(img, meta[resize_to][1:])

            imgs[i].append(img)

    return imgs


def write_location_images(loc, imgs, band, filters=False):
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

            np.save('cache/images/%s_%d_%d_%s.npy' % (loc, i, j, band), data[:, ys[i]:ys[i]+ysz, xs[j]:xs[j]+xsz])

    # Save debug location map
    if False:
        cv2.imwrite("%s.png" % loc, np.rollaxis((data - data.min()) * 255.0 / (data.max() - data.min()), 0, 3).astype(np.uint8))

    # Compute and save filters
    if filters:
        filter_data = np.zeros((1, ys[-1], xs[-1]), dtype=np.float32)
        filter_data[0] = sobel(np.clip(data[0] / 600.0, 0, 1)) + sobel(np.clip(data[1] / 600.0, 0, 1)) + sobel(np.clip(data[2] / 600.0, 0, 1))

        for i in xrange(n_location_images):
            for j in xrange(n_location_images):
                img = imgs[i][j]
                ysz = img.shape[1]
                xsz = img.shape[2]

                np.save('cache/images/%s_%d_%d_%sF.npy' % (loc, i, j, band), filter_data[:, ys[i]:ys[i]+ysz, xs[j]:xs[j]+xsz])


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

    write_location_images(loc, imgs_i, 'I', filters=True)
    write_location_images(loc, imgs_m, 'M')

    imgs_a = read_location_images(loc, 'sixteen_band', 'A', resize_to='shape_m')
    write_location_images(loc, imgs_a, 'A')

print "Done."
