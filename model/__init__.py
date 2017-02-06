import numpy as np
import cv2

from math import ceil

from util.meta import n_classes
from util import load_pickle, save_pickle


from keras.callbacks import ModelCheckpoint, Callback


patch_offset_range = 0.5  # Half of patch size
channel_shift_range = 0.0005


debug = False


band_size_factors = {
    'I': 1,
    'M': 4,
    'A': 4
}

band_n_channels = {
    'I': 3,
    'M': 8,
    'A': 8
}


class Monitor(Callback):

    def __init__(self, pipeline, image_ids):
        self.pipeline = pipeline
        self.image_ids = image_ids

    def on_epoch_end(self, epoch, logs={}):
        for image_id in self.image_ids:
            p = self.pipeline.predict(image_id)
            np.save('cache/preds/%s_%s.npy' % (image_id, self.pipeline.name), p)


class Input(object):

    def __init__(self, patch_size, band, downscale=1):
        self.band = band
        self.downscale = downscale
        self.patch_size = patch_size
        self.n_channels = band_n_channels[band]


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


class ModelPipeline(object):

    def __init__(self, name, arch, n_epoch, mask_patch_size, inputs, mask_downscale=1, classes=range(n_classes), batch_mode='grid', batch_size=64, epoch_batches=100):
        self.name = name
        self.arch = arch
        self.n_epoch = n_epoch

        self.inputs = dict((key, Input(mask_patch_size * mask_downscale / band_size_factors[inp['band']] / inp.get('downscale', 1), **inp)) for key, inp in inputs.items())

        self.mask_patch_size = mask_patch_size
        self.mask_downscale = mask_downscale

        self.n_patches = int(ceil(3400.0 / self.mask_patch_size / self.mask_downscale))

        self.classes = classes
        self.n_classes = len(classes)

        self.batch_mode = batch_mode
        self.batch_size = batch_size

        self.epoch_batches = epoch_batches

        # Initialize model
        input_shapes = dict((k, (i.n_channels, i.patch_size, i.patch_size)) for k, i in self.inputs.items())

        self.model = self.arch(input_shapes=input_shapes, n_classes=self.n_classes)

    def load(self):
        self.input_normalizers = load_pickle('cache/models/%s-norm.pickle' % self.name)
        self.model.load_weights('cache/models/%s.hdf5' % self.name)

    def fit(self, train_image_ids, val_image_ids=None):
        print "Fitting normalizers..."

        train_input_images = self.load_input_images(train_image_ids)

        self.fit_and_apply_normalizers(train_input_images)

        print "Training model with %d params..." % self.model.count_params()

        if self.batch_mode == 'grid':
            generator = self.grid_batch_generator(train_image_ids, train_input_images)
            n_samples = len(train_image_ids) * self.n_patches * self.n_patches
        elif self.batch_mode == 'random':
            generator = self.random_batch_generator(train_image_ids, train_input_images)
            n_samples = self.epoch_batches * self.batch_size

        callbacks = [ModelCheckpoint('cache/models/%s.hdf5' % self.name, monitor='loss', save_best_only=True)]

        if val_image_ids is not None:
            callbacks.append(Monitor(self, ['6100_2_2']))

        self.model.fit_generator(
            generator,
            samples_per_epoch=n_samples,
            nb_epoch=self.n_epoch, verbose=1,
            callbacks=callbacks, validation_data=None if val_image_ids is None else self.load_labelled_patches(val_image_ids))

    def fit_and_apply_normalizers(self, input_images):
        self.input_normalizers = {}
        for input_name, images in input_images.items():
            self.input_normalizers[input_name] = Normalizer().fit(images)

            for i in xrange(len(images)):
                images[i] = self.input_normalizers[input_name].transform(images[i])

        save_pickle('cache/models/%s-norm.pickle' % self.name, self.input_normalizers)

    def predict(self, image_id):
        meta = load_pickle('cache/meta/%s.pickle' % image_id)
        xbs = {}

        for input_name in self.inputs:
            inp = self.inputs[input_name]

            x = self.input_normalizers[input_name].transform(np.load('cache/images/%s_%s.npy' % (image_id, inp.band)))
            xb = np.zeros((self.n_patches * self.n_patches, x.shape[0], inp.patch_size, inp.patch_size), dtype=np.float32)

            k = 0
            for i in xrange(self.n_patches):
                for j in xrange(self.n_patches):
                    self.extract_patch(xb, x, k, i / (self.n_patches - 1.0), j / (self.n_patches - 1.0), inp.patch_size, inp.downscale)

                    k += 1

            xbs[input_name] = xb

        pb = self.model.predict(xbs)

        if debug:
            self.write_batch_images(xbs, pb, [(0, i / (self.n_patches - 1.0), j / (self.n_patches - 1.0)) for i in xrange(self.n_patches) for j in xrange(self.n_patches)], [image_id], 'pred')

        p = np.zeros((n_classes, meta['shape'][1], meta['shape'][2]), dtype=np.float32)
        c = np.zeros((n_classes, meta['shape'][1], meta['shape'][2]), dtype=np.float32)

        i_patch_step = (meta['shape'][1] - self.mask_patch_size*self.mask_downscale) / (self.n_patches - 1.0)
        j_patch_step = (meta['shape'][2] - self.mask_patch_size*self.mask_downscale) / (self.n_patches - 1.0)

        k = 0
        for i in xrange(self.n_patches):
            for j in xrange(self.n_patches):
                si = int(round(i*i_patch_step))
                sj = int(round(j*j_patch_step))

                p[self.classes, si:si+self.mask_patch_size*self.mask_downscale, sj:sj+self.mask_patch_size*self.mask_downscale] += pb[k] if self.mask_downscale == 1 else self.upscale_mask(pb[k])
                c[:, si:si+self.mask_patch_size*self.mask_downscale, sj:sj+self.mask_patch_size*self.mask_downscale] += 1

                k += 1

        return p / c

    def extract_patch(self, xx, x, k, oi, oj, patch_size, downscale):
        si = int(round(oi*(x.shape[1] - patch_size*downscale)))
        sj = int(round(oj*(x.shape[2] - patch_size*downscale)))

        if downscale == 1:
            xx[k] = x[:, si:si+patch_size, sj:sj+patch_size]
        else:
            for c in xrange(xx.shape[1]):
                xx[k, c] = cv2.resize(x[c, si:si+patch_size*downscale, sj:sj+patch_size*downscale].astype(np.float32), (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

    def upscale_mask(self, m):
        res = np.zeros((m.shape[0], m.shape[1]*self.mask_downscale, m.shape[2]*self.mask_downscale), dtype=np.float32)

        for c in xrange(m.shape[0]):
            res[c] = cv2.resize(m[c], (m.shape[1]*self.mask_downscale, m.shape[2]*self.mask_downscale), interpolation=cv2.INTER_LINEAR)

        return res

    def load_labelled_patches(self, image_ids):
        print "Loading image patches..."

        n = len(image_ids) * self.n_patches * self.n_patches

        yy = np.zeros((n, self.n_classes, self.mask_patch_size, self.mask_patch_size), dtype=np.float32)
        xx = {}
        for input_name, inp in self.inputs.items():
            xx[input_name] = np.zeros((n, inp.n_channels, inp.patch_size, inp.patch_size), dtype=np.float32)

        k = 0
        patches = []
        for img_idx, image_id in enumerate(image_ids):
            y = np.load('cache/masks/%s.npy' % image_id)[self.classes]
            x = {}
            for input_name, inp in self.inputs.items():
                x[input_name] = self.input_normalizers[input_name].transform(np.load('cache/images/%s_%s.npy' % (image_id, inp.band)))

            for i in xrange(self.n_patches):
                for j in xrange(self.n_patches):
                    oi = i / (self.n_patches - 1.0)
                    oj = j / (self.n_patches - 1.0)

                    self.extract_patch(yy, y, k, oi, oj, self.mask_patch_size, self.mask_downscale)

                    for input_name, inp in self.inputs.items():
                        self.extract_patch(xx[input_name], x[input_name], k, oi, oj, inp.patch_size, inp.downscale)

                    k += 1
                    patches.append((img_idx, oi, oj))

            if debug:
                self.write_batch_images(xx, yy, patches, image_ids, 'val')

        return xx, yy

    def augment_batch(self, x_batches, y_batch):
        for i in xrange(y_batch.shape[0]):
            if np.random.random() < 0.5:  # Mirror by x
                for x_batch in x_batches.values():
                    x_batch[i] = x_batch[i, :, ::-1, :]
                y_batch[i] = y_batch[i, :, ::-1, :]

            if np.random.random() < 0.5:  # Mirror by y
                for x_batch in x_batches.values():
                    x_batch[i] = x_batch[i, :, :, ::-1]
                y_batch[i] = y_batch[i, :, :, ::-1]

            if np.random.random() < 0.5:  # Rotate
                for x_batch in x_batches.values():
                    x_batch[i] = np.swapaxes(x_batch[i], 1, 2)
                y_batch[i] = np.swapaxes(y_batch[i], 1, 2)

            # Apply random channel shift
            for x_batch in x_batches.values():
                for c in xrange(x_batch.shape[1]):
                    x_batch[i, c] += np.random.uniform(-channel_shift_range, channel_shift_range)

    def write_batch_images(self, x_batches, y_batch, patches, image_ids, stage):
        for i, (img_idx, oi, oj) in enumerate(patches):
            if y_batch[0].sum() > 5 and np.random.random() < 0.01:
                for input_name, inp in self.inputs.items():
                    cv2.imwrite("debug/%s/%s_%3f_%3f_%s.png" % (stage, image_ids[img_idx], oi, oj, inp.band), np.rollaxis(np.clip(x_batches[input_name][i, :3], 0, 1) * 255.0, 0, 3).astype(np.uint8))
                cv2.imwrite("debug/%s/%s_%3f_%3f_mask.png" % (stage, image_ids[img_idx], oi, oj), np.rollaxis(np.clip(y_batch[i, [0, 1, 3]], 0, 1) * 255.0, 0, 3).astype(np.uint8))

    def load_input_images(self, image_ids):
        input_images = {}
        for input_name, inp in self.inputs.items():
            input_images[input_name] = [np.load('cache/images/%s_%s.npy' % (image_id, inp.band)) for image_id in image_ids]
        return input_images

    def grid_batch_generator(self, image_ids, input_images):
        masks = [np.load('cache/masks/%s.npy' % image_id)[self.classes] for image_id in image_ids]

        while True:
            # Prepare index of patch locations
            patches = []

            for img_idx in xrange(len(image_ids)):
                for i in xrange(self.n_patches):
                    for j in xrange(self.n_patches):
                        # Patch coord with random offset
                        oi = np.clip((i + np.random.uniform(-1, 1) * patch_offset_range) / (self.n_patches - 1.0), 0, 1)
                        oj = np.clip((j + np.random.uniform(-1, 1) * patch_offset_range) / (self.n_patches - 1.0), 0, 1)

                        # Add to patch list
                        patches.append((img_idx, oi, oj))

            # Shuffle index
            np.random.shuffle(patches)

            # Iterate over patches
            batch_start = 0
            while batch_start < len(patches):
                batch_patches = patches[batch_start:batch_start + self.batch_size]

                x_batches = {}
                for input_name, inp in self.inputs.items():
                    x_batches[input_name] = np.zeros((len(batch_patches), inp.n_channels, inp.patch_size, inp.patch_size), dtype=np.float32)
                y_batch = np.zeros((len(batch_patches), self.n_classes, self.mask_patch_size, self.mask_patch_size), dtype=np.float32)

                for i, (img_idx, oi, oj) in enumerate(batch_patches):
                    for input_name, inp in self.inputs.items():
                        self.extract_patch(x_batches[input_name], input_images[input_name][img_idx], i, oi, oi, inp.patch_size, inp.downscale)
                    self.extract_patch(y_batch, masks[img_idx], i, oi, oj, self.mask_patch_size, self.mask_downscale)

                self.augment_batch(x_batches, y_batch)

                # Write debug images
                if debug:
                    self.write_batch_images(x_batches, y_batch, patches, image_ids, 'train')

                yield x_batches, y_batch

                batch_start += self.batch_size

    def random_batch_generator(self, image_ids, input_images):
        threshold = 5

        masks = [np.load('cache/masks/%s.npy' % image_id)[self.classes] for image_id in image_ids]

        while True:
            x_batches = {}
            for input_name, inp in self.inputs.items():
                x_batches[input_name] = np.zeros((self.batch_size, inp.n_channels, inp.patch_size, inp.patch_size), dtype=np.float32)

            y_batch = np.zeros((self.batch_size, self.n_classes, self.mask_patch_size, self.mask_patch_size), dtype=np.float32)

            # Extract batch patches
            k = 0
            patches = []
            while k < self.batch_size:
                img_idx = np.random.randint(len(masks))

                oi = np.random.uniform(0, 1)
                oj = np.random.uniform(0, 1)

                self.extract_patch(y_batch, masks[img_idx], k, oi, oj, self.mask_patch_size, self.mask_downscale)

                if y_batch[k].sum() < threshold:
                    continue

                for input_name, inp in self.inputs.items():
                    self.extract_patch(x_batches[input_name], input_images[input_name][img_idx], k, oi, oj, inp.patch_size, inp.downscale)

                patches.append((img_idx, oi, oj))
                k += 1

            # Augment them
            self.augment_batch(x_batches, y_batch)

            # Write debug images
            if debug:
                self.write_batch_images(x_batches, y_batch, patches, image_ids, 'train')

            yield x_batches, y_batch
