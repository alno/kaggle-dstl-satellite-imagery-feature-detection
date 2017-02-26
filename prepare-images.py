from util.meta import locations, image_border
from util import load_pickle, save_pickle

from skimage.filters import sobel
from joblib import Parallel, delayed

import tifffile as tiff
import numpy as np
import cv2

n_location_images = 5


def normalize(src):
    dst = np.empty(shape=src.shape, dtype=np.float32)

    for c in xrange(src.shape[0]):
        l, h = np.percentile(src[c], [1, 99])

        dst[c] = (src[c].astype(np.float64) - l) / (h - l)

    return dst


def resize(src, shape):
    dst = np.empty(shape=(src.shape[0], shape[0], shape[1]))

    for c in xrange(src.shape[0]):
        dst[c] = cv2.resize(src[c], (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

    return dst


def add_border(src):
    dst = np.empty(shape=(src.shape[0], src.shape[1] + 2 * image_border, src.shape[2] + 2 * image_border))

    for c in xrange(src.shape[0]):
        dst[c] = cv2.copyMakeBorder(src[c], top=image_border, bottom=image_border, left=image_border, right=image_border, borderType=cv2.BORDER_REPLICATE)

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

            if len(img.shape) == 2:
                img = img[np.newaxis, :, :]

            if resize_to is not None:
                meta = load_pickle('cache/meta/%s_%d_%d.pickle' % (loc, i, j))
                img = resize(img, meta[resize_to][1:])

            imgs[i].append(img)

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
            data[:, ys[i]:ys[i+1], xs[j]:xs[j+1]] = imgs[i][j]

    # Add border by replicating image data
    data = add_border(data)

    return data, xs, ys


def write_location_images(loc, data, xs, ys, band, filters=False):
    # Save images
    for i in xrange(n_location_images):
        for j in xrange(n_location_images):
            np.save('cache/images/%s_%d_%d_%s.npy' % (loc, i, j, band), data[:, ys[i]:ys[i+1] + 2 * image_border, xs[j]:xs[j+1] + 2 * image_border])

    # Save debug location map
    if False:
        cv2.imwrite("%s.png" % loc, np.rollaxis((data - data.min()) * 255.0 / (data.max() - data.min()), 0, 3).astype(np.uint8))


def compute_filters(data):
    filter_data = np.zeros((1, data.shape[1], data.shape[2]), dtype=np.float32)
    filter_data[0] = sobel(np.clip(data[0] / 600.0, 0, 1)) + sobel(np.clip(data[1] / 600.0, 0, 1)) + sobel(np.clip(data[2] / 600.0, 0, 1))

    return filter_data


def compute_indices(m):
    eps = 1e-3

    blue = m[1].astype(np.float64)
    green = m[2].astype(np.float64)
    red = m[4].astype(np.float64)

    nir1 = m[6].astype(np.float64)

    indices_data = np.zeros((3, m.shape[1], m.shape[2]), dtype=np.float32)
    indices_data[0] = (green - nir1) / (nir1 + green + eps)  # ndwi - nir1, green
    indices_data[1] = (nir1 - red) / (nir1 + red + eps)  # ndvi - nir1, red
    indices_data[2] = (nir1 - blue) / (nir1 + blue + eps)  # bai - nir1, blue

    return indices_data


def prepare_location(loc):
    print "  Processing %s..." % loc

    data_i, xs_i, ys_i = read_location_images(loc, 'three_band')
    data_m, xs_m, ys_m = read_location_images(loc, 'sixteen_band', 'M')
    #data_p, xs_p, ys_p = read_location_images(loc, 'sixteen_band', 'P')

    # Prepare images
    for i in xrange(n_location_images):
        for j in xrange(n_location_images):
            meta = {
                'shape': (0, ys_i[i+1] - ys_i[i], xs_i[j+1] - xs_i[j]),
                'shape_i': (data_i.shape[0], ys_i[i+1] - ys_i[i], xs_i[j+1] - xs_i[j]),
                'shape_m': (data_m.shape[0], ys_m[i+1] - ys_m[i], xs_m[j+1] - xs_m[j]),
                #'shape_p': (data_p.shape[0], ys_p[i+1] - ys_p[i], xs_p[j+1] - xs_p[j])
            }

            save_pickle('cache/meta/%s_%d_%d.pickle' % (loc, i, j), meta)

    write_location_images(loc, data_i, xs_i, ys_i, 'I')
    write_location_images(loc, data_m, xs_m, ys_m, 'M')
    #write_location_images(loc, data_p, xs_p, ys_p, 'P')

    write_location_images(loc, normalize(data_m), xs_m, ys_m, 'MN')  # Write location-normalized M channels

    write_location_images(loc, compute_filters(data_i), xs_i, ys_i, 'IF')
    write_location_images(loc, compute_indices(data_m), xs_m, ys_m, 'MI')

    #data_a, xs_a, ys_a = read_location_images(loc, 'sixteen_band', 'A', resize_to='shape_m')

    #write_location_images(loc, data_a, xs_a, ys_a, 'A')


print "Preparing image data..."

# Prepare locations
Parallel(n_jobs=2)(delayed(prepare_location)(loc) for loc in locations)

print "Done."
