import numpy as np
import cv2

from math import ceil

from util.meta import n_classes, image_border
from util import load_pickle, save_pickle

from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam

from .objectives import combined_loss, jaccard_coef, jaccard_coef_int
from .ema import ExponentialMovingAverage

patch_offset_range = 0.5
round_offsets = True

debug = False


band_size_factors = {
    'I': 1,
    'IF': 1,
    'M': 4,
    'MN': 4,
    'MI': 4,
    'A': 4
}

band_n_channels = {
    'I': 3,
    'IF': 1,
    'M': 8,
    'MN': 8,
    'MI': 3,
    'A': 8
}


class_priors = np.array([
    0.4,
    0.1,
    0.2,
    0.2,
    0.3,
    0.5,
    0.1,
    0.05,
    0.01,
    0.01,
])

class_smooths = np.array([
    80,
    40,
    40,
    80,
    150,
    200,
    30,
    15,
    10,
    15,
])


def upscale_mask(m, downscale):
    res = np.zeros((m.shape[0], m.shape[1]*downscale, m.shape[2]*downscale), dtype=np.float32)

    for c in xrange(m.shape[0]):
        res[c] = cv2.resize(m[c], (m.shape[1]*downscale, m.shape[2]*downscale), interpolation=cv2.INTER_LINEAR)

    return res


def extract_patch(xx, x, k, oi, oj, patch_size, downscale):
    si = int(round(oi*(x.shape[1] - 2*image_border - patch_size*downscale))) + image_border
    sj = int(round(oj*(x.shape[2] - 2*image_border - patch_size*downscale))) + image_border

    if downscale == 1:
        xx[k] = x[:, si:si+patch_size, sj:sj+patch_size]
    else:
        for c in xrange(xx.shape[1]):
            xx[k, c] = cv2.resize(x[c, si:si+patch_size*downscale, sj:sj+patch_size*downscale].astype(np.float32), (patch_size, patch_size), interpolation=cv2.INTER_AREA)


class Augmenter(object):

    def __init__(self, channel_shift_range=0.0005, channel_scale_range=0.0001, mirror=True, transpose=True, rotation=0, scale=0):
        self.mirror = mirror
        self.transpose = transpose
        self.rotation = rotation
        self.scale = scale
        self.channel_shift_range = channel_shift_range
        self.channel_scale_range = channel_scale_range

    def augment_batch(self, x_batches, y_batch):
        for i in xrange(y_batch.shape[0]):
            if self.rotation > 0 or self.scale > 0:
                theta = np.random.uniform(-self.rotation, self.rotation)
                scale = np.random.uniform(-self.scale, self.scale) + 1

                for x_batch in x_batches.values():
                    w, h = x_batch.shape[3], x_batch.shape[2]
                    transform = cv2.getRotationMatrix2D((w/2, h/2), theta, scale)

                    for c in xrange(x_batch.shape[1]):
                        x_batch[i, c] = cv2.warpAffine(x_batch[i, c], transform, dsize=(w, h), flags=cv2.INTER_LINEAR)

                w, h = y_batch.shape[3], y_batch.shape[2]
                transform = cv2.getRotationMatrix2D((w/2, h/2), theta, scale)
                for c in xrange(y_batch.shape[1]):
                    y_batch[i, c] = cv2.warpAffine(y_batch[i, c], transform, dsize=(w, h), flags=cv2.INTER_LINEAR)

            if self.mirror and np.random.random() < 0.5:  # Mirror by x
                for x_batch in x_batches.values():
                    x_batch[i] = x_batch[i, :, ::-1, :]
                y_batch[i] = y_batch[i, :, ::-1, :]

            if self.mirror and np.random.random() < 0.5:  # Mirror by y
                for x_batch in x_batches.values():
                    x_batch[i] = x_batch[i, :, :, ::-1]
                y_batch[i] = y_batch[i, :, :, ::-1]

            if self.transpose and np.random.random() < 0.5:  # Transpose
                for x_batch in x_batches.values():
                    x_batch[i] = np.swapaxes(x_batch[i], 1, 2)
                y_batch[i] = np.swapaxes(y_batch[i], 1, 2)

            # Apply random channel scale and shift
            for input_name, x_batch in x_batches.items():
                if 'F' in input_name:  # Don't augment channels for filters input
                    continue

                for c in xrange(x_batch.shape[1]):
                    if self.channel_scale_range > 0:
                        x_batch[i, c] *= np.random.uniform(1-self.channel_scale_range, 1+self.channel_scale_range)

                    if self.channel_shift_range > 0:
                        x_batch[i, c] += np.random.uniform(-self.channel_shift_range, self.channel_shift_range)


class Validator(Callback):

    def __init__(self, pipeline, image_ids):
        self.pipeline = pipeline
        self.image_ids = image_ids
        self.image_masks = dict((image_id, np.load('cache/masks/%s.npy' % image_id)) for image_id in image_ids)

    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1) % 5 != 0:
            return

        print
        print "  Validating epoch %d.." % (epoch+1)

        class_intersections = np.zeros(n_classes, dtype=np.float64)
        class_unions = np.zeros(n_classes, dtype=np.float64) + 1e-5

        class_intersections_int = np.zeros(n_classes, dtype=np.float64)
        class_unions_int = np.zeros(n_classes, dtype=np.float64) + 1e-5

        for image_id in self.image_ids:
            mask = self.image_masks[image_id].astype(np.float64)

            pred = self.pipeline.predict(image_id).astype(np.float64)
            pred_int = pred > 0.5

            np.save('cache/preds/%s_%s.npy' % (image_id, self.pipeline.name), pred)

            inter = (pred * mask).sum(axis=(1, 2))
            union = (pred + mask).sum(axis=(1, 2)) - inter

            inter_int = (pred_int * mask).sum(axis=(1, 2))
            union_int = (pred_int + mask).sum(axis=(1, 2)) - inter_int

            class_intersections += inter
            class_unions += union

            class_intersections_int += inter_int
            class_unions_int += union_int

        class_jacs = class_intersections / class_unions
        class_jacs_int = class_intersections_int / class_unions_int

        print "  Class jac: [%s], mean jac: %s" % (' '.join('%.5f' % j for j in class_jacs), class_jacs.mean())
        print "  Class jac_int: [%s], mean jac_int: %s" % (' '.join('%.5f' % j for j in class_jacs_int), class_jacs_int.mean())


class Input(object):

    def __init__(self, patch_size, band, downscale=1):
        self.band = band
        self.downscale = downscale
        self.patch_size = patch_size
        self.n_channels = band_n_channels[band]

    def round_offsets(self, oi, oj, img):
        ni = img.shape[1] - self.patch_size * self.downscale
        nj = img.shape[2] - self.patch_size * self.downscale

        oi = round(oi * ni) / ni
        oj = round(oj * nj) / nj

        return oi, oj


class Normalizer(object):

    def fit(self, images):
        self.mins = np.empty(images[0].shape[0])
        self.maxs = np.empty(images[0].shape[0])

        for c in xrange(images[0].shape[0]):
            self.mins[c] = min(img[c].min() for img in images)
            self.maxs[c] = max(np.percentile(img[c], 99) for img in images)

        return self

    def transform(self, img):
        res = np.empty(img.shape, dtype=np.float32)

        for c in xrange(img.shape[0]):
            res[c] = (img[c] - self.mins[c]) / (self.maxs[c] - self.mins[c])

        return res


class MeanStdNormalizer(object):

    def fit(self, images):
        self.means = np.empty(images[0].shape[0])
        self.stds = np.empty(images[0].shape[0])

        for c in xrange(images[0].shape[0]):
            vals = np.hstack(img[c].flatten() for img in images)

            self.means[c] = vals.mean()
            self.stds[c] = vals.std()

        return self

    def transform(self, img):
        res = np.empty(img.shape, dtype=np.float32)

        for c in xrange(img.shape[0]):
            res[c] = (img[c] - self.means[c]) / self.stds[c]

        return res


class ModelPipeline(object):

    def __init__(self, name, arch, mask_patch_size, inputs, mask_downscale=1, classes=range(n_classes), arch_options={}, normalization='minmax'):
        self.name = name

        self.inputs = dict((key, Input(mask_patch_size * mask_downscale / band_size_factors[inp['band']] / inp.get('downscale', 1), **inp)) for key, inp in inputs.items())
        self.coarse_input = max(self.inputs.keys(), key=lambda inp: band_size_factors[self.inputs[inp].band])

        self.mask_patch_size = mask_patch_size
        self.mask_downscale = mask_downscale

        self.n_patches = int(ceil(3400.0 / (self.mask_patch_size - 16) / self.mask_downscale))

        self.classes = classes
        self.n_classes = len(classes)

        self.normalization = normalization

        # Initialize model
        input_shapes = dict((k, (i.n_channels, i.patch_size, i.patch_size)) for k, i in self.inputs.items())

        self.model = arch(input_shapes=input_shapes, n_classes=self.n_classes, **arch_options)

    def load(self):
        self.input_normalizers = load_pickle('cache/models/%s-norm.pickle' % self.name)
        self.load_weights(self.name)

    def load_weights(self, name):
        self.model.load_weights('cache/models/%s.hdf5' % name)

    def fit(self, train_image_ids, val_image_ids=None, n_epoch=100, epoch_batches='grid', batch_size=64, augment={}, optimizer=None, loss_jac_weight=0.1, batch_class_threshold=0, class_weights=1.0, ema=False, batch_noclass_accept_proba=0, batch_noclass_accept_proba_growth=0):
        print "Fitting normalizers..."

        augmenter = Augmenter(**augment)

        train_input_images = self.load_input_images(train_image_ids)
        train_masks = self.load_masks(train_image_ids)

        self.fit_and_apply_normalizers(train_input_images)

        print "Preparing batch generators..."

        if epoch_batches == 'grid':
            generator = self.grid_batch_generator(train_image_ids, train_input_images, train_masks, augmenter=augmenter, batch_size=batch_size)
            n_samples = len(train_image_ids) * self.n_patches * self.n_patches
        else:
            generator = self.random_batch_generator(train_image_ids, train_input_images, train_masks, augmenter=augmenter, batch_size=batch_size, batch_class_threshold=batch_class_threshold, batch_noclass_accept_proba=batch_noclass_accept_proba, batch_noclass_accept_proba_growth=batch_noclass_accept_proba_growth)
            n_samples = epoch_batches * batch_size

        print "Training model with %d params..." % self.model.count_params()

        callbacks = [
            ExponentialMovingAverage(decay=0.995, filepath='cache/models/%s.hdf5' % self.name) if ema else ModelCheckpoint('cache/models/%s.hdf5' % self.name, monitor='loss', save_best_only=False, save_weights_only=True),
        ]

        if val_image_ids is not None:
            callbacks.append(Validator(self, val_image_ids))

        def loss(y, p):
            return combined_loss(y, p, class_smooths[self.classes], class_priors[self.classes], jac_weight=loss_jac_weight, class_weights=class_weights)

        def jac(y, p):
            return jaccard_coef(y, p, class_weights=class_weights)

        def jac_int(y, p):
            return jaccard_coef_int(y, p, class_weights=class_weights)

        if optimizer is None:
            optimizer = Adam(3e-3, decay=4e-4)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=[jac, jac_int])

        self.model.fit_generator(
            generator,
            samples_per_epoch=n_samples,
            nb_epoch=n_epoch, verbose=1,
            callbacks=callbacks)
        self.model.save_weights('cache/models/%s.hdf5' % self.name)

    def fit_and_apply_normalizers(self, input_images):
        self.input_normalizers = {}
        for input_name, images in input_images.items():
            if self.normalization == 'minmax':
                norm = Normalizer()
            elif self.normalization == 'std':
                norm = MeanStdNormalizer()
            else:
                raise ValueError("Unknown normalization: %s" % self.normalization)

            self.input_normalizers[input_name] = norm.fit(images)

            for i in xrange(len(images)):
                images[i] = self.input_normalizers[input_name].transform(images[i])

        save_pickle('cache/models/%s-norm.pickle' % self.name, self.input_normalizers)

    def predict(self, image_id):
        meta = load_pickle('cache/meta/%s.pickle' % image_id)
        xbs = {}

        x = {}
        for input_name, inp in self.inputs.items():
            x[input_name] = self.input_normalizers[input_name].transform(np.load('cache/images/%s_%s.npy' % (image_id, inp.band)))

        for input_name, inp in self.inputs.items():
            xb = np.zeros((self.n_patches * self.n_patches, x[input_name].shape[0], inp.patch_size, inp.patch_size), dtype=np.float32)

            k = 0
            for i in xrange(self.n_patches):
                for j in xrange(self.n_patches):
                    oi = i / (self.n_patches - 1.0)
                    oj = j / (self.n_patches - 1.0)

                    if round_offsets:
                        oi, oj = self.inputs[self.coarse_input].round_offsets(oi, oj, x[self.coarse_input])

                    extract_patch(xb, x[input_name], k, oi, oj, inp.patch_size, inp.downscale)

                    k += 1

            xbs[input_name] = xb

        pb = self.model.predict(xbs, batch_size=32)

        if debug:
            self.write_batch_images(xbs, pb, [(0, i / (self.n_patches - 1.0), j / (self.n_patches - 1.0)) for i in xrange(self.n_patches) for j in xrange(self.n_patches)], [image_id], 'pred')

        p = np.zeros((n_classes, meta['shape'][1], meta['shape'][2]), dtype=np.float32)
        c = np.zeros((n_classes, meta['shape'][1], meta['shape'][2]), dtype=np.float32)

        k = 0
        for i in xrange(self.n_patches):
            for j in xrange(self.n_patches):
                oi = i / (self.n_patches - 1.0)
                oj = j / (self.n_patches - 1.0)

                if round_offsets:
                    oi, oj = self.inputs[self.coarse_input].round_offsets(oi, oj, x[self.coarse_input])

                si = int(round(oi * (meta['shape'][1] - self.mask_patch_size*self.mask_downscale)))
                sj = int(round(oj * (meta['shape'][2] - self.mask_patch_size*self.mask_downscale)))

                p[self.classes, si:si+self.mask_patch_size*self.mask_downscale, sj:sj+self.mask_patch_size*self.mask_downscale] += pb[k] if self.mask_downscale == 1 else upscale_mask(pb[k], self.mask_downscale)
                c[:, si:si+self.mask_patch_size*self.mask_downscale, sj:sj+self.mask_patch_size*self.mask_downscale] += 1

                k += 1

        return p / c

    def load_input_images(self, image_ids):
        input_images = {}
        for input_name, inp in self.inputs.items():
            input_images[input_name] = [np.load('cache/images/%s_%s.npy' % (image_id, inp.band)) for image_id in image_ids]
        return input_images

    def load_masks(self, image_ids):
        masks = []
        for image_id in image_ids:
            mask = np.load('cache/masks/%s.npy' % image_id)

            masks.append(np.zeros((self.n_classes, mask.shape[1] + 2 * image_border, mask.shape[2] + 2 * image_border), dtype=mask.dtype))
            masks[-1][:, image_border:mask.shape[1] + image_border, image_border:mask.shape[2] + image_border] = mask[self.classes]

        return masks

    def write_batch_images(self, x_batches, y_batch, patches, image_ids, stage):
        for i, (img_idx, oi, oj) in enumerate(patches):
            if y_batch[0].sum() > 5 and np.random.random() < 0.01:
                for input_name, inp in self.inputs.items():
                    cv2.imwrite("debug/%s/%s_%3f_%3f_%s.png" % (stage, image_ids[img_idx], oi, oj, inp.band), np.rollaxis(np.clip(x_batches[input_name][i, :3], 0, 1) * 255.0, 0, 3).astype(np.uint8))
                cv2.imwrite("debug/%s/%s_%3f_%3f_mask.png" % (stage, image_ids[img_idx], oi, oj), np.rollaxis(np.clip(y_batch[i, [0, 1, 3]], 0, 1) * 255.0, 0, 3).astype(np.uint8))

    def grid_batch_generator(self, image_ids, input_images, masks, augmenter, batch_size):
        while True:
            # Prepare index of patch locations
            patches = []

            for img_idx in xrange(len(image_ids)):
                for i in xrange(self.n_patches):
                    for j in xrange(self.n_patches):
                        # Patch coord with random offset
                        oi = np.clip((i + np.random.uniform(-1, 1) * patch_offset_range) / (self.n_patches - 1.0), 0, 1)
                        oj = np.clip((j + np.random.uniform(-1, 1) * patch_offset_range) / (self.n_patches - 1.0), 0, 1)

                        if round_offsets:
                            oi, oj = self.inputs[self.coarse_input].round_offsets(oi, oj, input_images[self.coarse_input][img_idx])

                        # Add to patch list
                        patches.append((img_idx, oi, oj))

            # Shuffle index
            np.random.shuffle(patches)

            # Iterate over patches
            batch_start = 0
            while batch_start < len(patches):
                batch_patches = patches[batch_start:batch_start + batch_size]

                x_batches = {}
                for input_name, inp in self.inputs.items():
                    x_batches[input_name] = np.zeros((len(batch_patches), inp.n_channels, inp.patch_size, inp.patch_size), dtype=np.float32)
                y_batch = np.zeros((len(batch_patches), self.n_classes, self.mask_patch_size, self.mask_patch_size), dtype=np.float32)

                for i, (img_idx, oi, oj) in enumerate(batch_patches):
                    for input_name, inp in self.inputs.items():
                        extract_patch(x_batches[input_name], input_images[input_name][img_idx], i, oi, oi, inp.patch_size, inp.downscale)
                    extract_patch(y_batch, masks[img_idx], i, oi, oj, self.mask_patch_size, self.mask_downscale)

                augmenter.augment_batch(x_batches, y_batch)

                # Write debug images
                if debug:
                    self.write_batch_images(x_batches, y_batch, patches, image_ids, 'train')

                yield x_batches, y_batch

                batch_start += batch_size

    def random_batch_generator(self, image_ids, input_images, masks, augmenter, batch_size, batch_class_threshold, batch_noclass_accept_proba, batch_noclass_accept_proba_growth):
        while True:
            x_batches = {}
            for input_name, inp in self.inputs.items():
                x_batches[input_name] = np.zeros((batch_size, inp.n_channels, inp.patch_size, inp.patch_size), dtype=np.float32)

            y_batch = np.zeros((batch_size, self.n_classes, self.mask_patch_size, self.mask_patch_size), dtype=np.float32)

            # Extract batch patches
            k = 0
            patches = []
            while k < batch_size:
                img_idx = np.random.randint(len(masks))

                oi = np.random.uniform(0, 1)
                oj = np.random.uniform(0, 1)

                if round_offsets:
                    oi, oj = self.inputs[self.coarse_input].round_offsets(oi, oj, input_images[self.coarse_input][img_idx])

                extract_patch(y_batch, masks[img_idx], k, oi, oj, self.mask_patch_size, self.mask_downscale)

                # Skip image if it doesn't pass threshold and random acceptance
                if all(y_batch[k].sum(axis=(1, 2)) < batch_class_threshold) and np.random.rand() > batch_noclass_accept_proba:
                    continue

                for input_name, inp in self.inputs.items():
                    extract_patch(x_batches[input_name], input_images[input_name][img_idx], k, oi, oj, inp.patch_size, inp.downscale)

                patches.append((img_idx, oi, oj))
                k += 1

            # Augment them
            augmenter.augment_batch(x_batches, y_batch)

            # Write debug images
            if debug:
                self.write_batch_images(x_batches, y_batch, patches, image_ids, 'train')

            yield x_batches, y_batch

            batch_noclass_accept_proba += batch_noclass_accept_proba_growth
