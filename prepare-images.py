from util.meta import locations, image_size

import tifffile as tiff
import numpy as np
import cv2


print "Preparing image data..."

# Prepare location
for loc in locations:
    print "  Processing %s..." % loc

    data = np.zeros((3, 5 * image_size, 5 * image_size), dtype=np.float32)

    # Prepare images
    for i in xrange(5):
        for j in xrange(5):
            img = tiff.imread('../input/three_band/%s_%d_%d.tif' % (loc, i, j))

            for c in xrange(3):
                data[c, i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size] = cv2.resize(img[c, :, :], (image_size, image_size), interpolation=cv2.INTER_CUBIC)

    # Save images
    for i in xrange(5):
        for j in xrange(5):
            np.save('cache/images/%s_%d_%d.npy' % (loc, i, j), data[:, i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size])

print "Done."
