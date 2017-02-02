import numpy as np
import cv2

from math import ceil

from util.meta import n_classes


from keras.callbacks import ModelCheckpoint, Callback


patch_offset_range = 0.5  # Half of patch size
channel_shift_range = 0.05


debug = False


class Monitor(Callback):

    def __init__(self, pipeline, image_ids):
        self.pipeline = pipeline
        self.image_ids = image_ids

    def on_epoch_end(self, epoch, logs={}):
        for image_id in self.image_ids:
            p = self.pipeline.predict(image_id)
            np.save('cache/preds/%s.npy' % image_id, p)


class ModelPipeline(object):

    def __init__(self, name, arch, n_epoch, patch_size, downscale, classes=range(n_classes), batch_mode='grid'):
        self.name = name
        self.arch = arch
        self.n_epoch = n_epoch

        self.patch_size = patch_size
        self.downscale = downscale

        self.n_patches = int(ceil(3400.0 / patch_size / downscale))
        self.n_channels = 3

        self.classes = classes
        self.n_classes = len(classes)

        self.batch_mode = batch_mode
        self.batch_size = 64

        self.random_batch_per_epoch = 100

        self.model = self.arch(input_shape=(self.n_channels, self.patch_size, self.patch_size), n_classes=self.n_classes)

    def load(self):
        self.model.load_weights('cache/models/%s.hdf5' % self.name)

    def fit(self, train_image_ids, val_image_ids):
        print "Training model with %d params..." % self.model.count_params()

        if self.batch_mode == 'grid':
            generator = self.grid_batch_generator(train_image_ids)
            n_samples = len(train_image_ids) * self.n_patches * self.n_patches
        elif self.batch_mode == 'random':
            generator = self.random_batch_generator(train_image_ids)
            n_samples = self.random_batch_per_epoch * self.batch_size

        callbacks = [ModelCheckpoint('cache/models/%s.hdf5' % self.name, monitor='loss', save_best_only=True), Monitor(self, ['6100_2_2'])]

        self.model.fit_generator(
            generator,
            samples_per_epoch=n_samples,
            nb_epoch=self.n_epoch, verbose=1,
            callbacks=callbacks, validation_data=None if val_image_ids is None else self.load_labelled_patches(val_image_ids))

    def predict(self, image_id):
        x = np.load('cache/images/%s.npy' % image_id)

        i_patch_step = (x.shape[1] - self.patch_size*self.downscale) / (self.n_patches - 1.0)
        j_patch_step = (x.shape[2] - self.patch_size*self.downscale) / (self.n_patches - 1.0)
        xb = np.zeros((self.n_patches * self.n_patches, x.shape[0], self.patch_size, self.patch_size), dtype=np.float16)

        k = 0
        for i in xrange(self.n_patches):
            for j in xrange(self.n_patches):
                si = int(round(i*i_patch_step))
                sj = int(round(j*j_patch_step))

                self.extract_patch(xb, x, k, si, sj)

                k += 1

        pb = self.model.predict(xb)

        p = np.zeros((n_classes, x.shape[1], x.shape[2]), dtype=np.float32)
        c = np.zeros((n_classes, x.shape[1], x.shape[2]), dtype=np.float32)

        k = 0
        for i in xrange(self.n_patches):
            for j in xrange(self.n_patches):
                si = int(round(i*i_patch_step))
                sj = int(round(j*j_patch_step))

                p[self.classes, si:si+self.patch_size*self.downscale, sj:sj+self.patch_size*self.downscale] += pb[k] if self.downscale == 1 else self.upscale(pb[k])
                c[:, si:si+self.patch_size*self.downscale, sj:sj+self.patch_size*self.downscale] += 1

                k += 1

        return p / c

    def extract_patch(self, xx, x, k, si, sj):
        if self.downscale == 1:
            xx[k] = x[:, si:si+self.patch_size, sj:sj+self.patch_size]
        else:
            for c in xrange(xx.shape[1]):
                xx[k, c] = cv2.resize(x[c, si:si+self.patch_size*self.downscale, sj:sj+self.patch_size*self.downscale].astype(np.float32), (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)

    def upscale(self, m):
        res = np.zeros((m.shape[0], m.shape[1]*self.downscale, m.shape[2]*self.downscale), dtype=np.float32)

        for c in xrange(m.shape[0]):
            res[c] = cv2.resize(m[c], (m.shape[1]*self.downscale, m.shape[2]*self.downscale), interpolation=cv2.INTER_LINEAR)

        return res

    def load_labelled_patches(self, image_ids):
        print "Loading image patches..."

        n = len(image_ids) * self.n_patches * self.n_patches

        xx = np.zeros((n, self.n_channels, self.patch_size, self.patch_size), dtype=np.float32)
        yy = np.zeros((n, self.n_classes, self.patch_size, self.patch_size), dtype=np.uint8)

        k = 0
        for image_id in image_ids:
            x = np.load('cache/images/%s.npy' % image_id)
            y = np.load('cache/masks/%s.npy' % image_id)[self.classes]

            i_patch_step = (x.shape[1] - self.patch_size*self.downscale) / (self.n_patches - 1.0)
            j_patch_step = (x.shape[2] - self.patch_size*self.downscale) / (self.n_patches - 1.0)

            for i in xrange(self.n_patches):
                for j in xrange(self.n_patches):
                    si = int(round(i*i_patch_step))
                    sj = int(round(j*j_patch_step))

                    self.extract_patch(xx, x, k, si, sj)
                    self.extract_patch(yy, y, k, si, sj)

                    k += 1

        return xx, yy

    def augment_batch(self, x_batch, y_batch):
        for i in xrange(x_batch.shape[0]):
            if np.random.random() < 0.5:  # Mirror by x
                x_batch[i] = x_batch[i, :, ::-1, :]
                y_batch[i] = y_batch[i, :, ::-1, :]

            if np.random.random() < 0.5:  # Mirror by y
                x_batch[i] = x_batch[i, :, :, ::-1]
                y_batch[i] = y_batch[i, :, :, ::-1]

            if np.random.random() < 0.5:  # Rotate
                x_batch[i] = np.swapaxes(x_batch[i], 1, 2)
                y_batch[i] = np.swapaxes(y_batch[i], 1, 2)

            # Apply random channel shift
            for c in xrange(x_batch.shape[1]):
                x_batch[i, c] += np.random.uniform(-channel_shift_range, channel_shift_range)

    def grid_batch_generator(self, image_ids):
        src_path_size = self.patch_size*self.downscale

        images = [np.load('cache/images/%s.npy' % image_id) for image_id in image_ids]
        masks = [np.load('cache/masks/%s.npy' % image_id)[self.classes] for image_id in image_ids]

        while True:
            # Prepare index of patch locations
            patches = []

            for img_idx in xrange(len(image_ids)):
                x = images[img_idx]

                i_patch_step = (x.shape[1] - src_path_size) / (self.n_patches - 1.0)
                j_patch_step = (x.shape[2] - src_path_size) / (self.n_patches - 1.0)

                for i in xrange(self.n_patches):
                    for j in xrange(self.n_patches):
                        # Patch coord with random offset
                        si = i*i_patch_step + np.random.uniform(-1, 1)*src_path_size*patch_offset_range
                        sj = j*j_patch_step + np.random.uniform(-1, 1)*src_path_size*patch_offset_range

                        # Clip and round offsets
                        si = int(round(np.clip(si, 0, x.shape[1] - src_path_size)))
                        sj = int(round(np.clip(sj, 0, x.shape[2] - src_path_size)))

                        # Add to patch list
                        patches.append((img_idx, si, sj))

            # Shuffle index
            np.random.shuffle(patches)

            # Iterate over patches
            batch_start = 0
            while batch_start < len(patches):
                batch_patches = patches[batch_start:batch_start + self.batch_size]

                x_batch = np.zeros((len(batch_patches), self.n_channels, self.patch_size, self.patch_size), dtype=np.float32)
                y_batch = np.zeros((len(batch_patches), self.n_classes, self.patch_size, self.patch_size), dtype=np.float32)

                for i, (img_idx, si, sj) in enumerate(batch_patches):
                    self.extract_patch(x_batch, images[img_idx], i, si, sj)
                    self.extract_patch(y_batch, masks[img_idx], i, si, sj)

                self.augment_batch(x_batch, y_batch)

                # Write debug images
                if debug:
                    for i in xrange(len(batch_patches)):
                        if np.random.random() < 0.01:
                            cv2.imwrite("debug/image_%s_%d_%d.png" % (image_ids[img_idx], si, sj), np.rollaxis(np.clip(x_batch[i], 0, 1) * 255.0, 0, 3).astype(np.uint8))
                            cv2.imwrite("debug/mask_%s_%d_%d.png" % (image_ids[img_idx], si, sj), np.rollaxis(np.clip(y_batch[i, [0, 3, 4]], 0, 1) * 255.0, 0, 3).astype(np.uint8))

                yield x_batch, y_batch

                batch_start += self.batch_size

    def random_batch_generator(self, image_ids):
        threshold = 4
        src_patch_size = self.patch_size*self.downscale

        images = [np.load('cache/images/%s.npy' % image_id) for image_id in image_ids]
        masks = [np.load('cache/masks/%s.npy' % image_id)[self.classes] for image_id in image_ids]

        while True:
            x_batch = np.zeros((self.batch_size, self.n_channels, self.patch_size, self.patch_size), dtype=np.float32)
            y_batch = np.zeros((self.batch_size, self.n_classes, self.patch_size, self.patch_size), dtype=np.float32)

            # Extract batch patches
            i = 0
            while i < self.batch_size:
                img_idx = np.random.randint(len(images))

                x = images[img_idx]
                y = masks[img_idx]

                si = int(round(np.random.uniform(0, x.shape[1] - src_patch_size)))
                sj = int(round(np.random.uniform(0, x.shape[2] - src_patch_size)))

                if y[:, si:si+src_patch_size, sj:sj+src_patch_size].sum() < threshold:
                    continue

                self.extract_patch(x_batch, x, i, si, sj)
                self.extract_patch(y_batch, y, i, si, sj)

                i += 1

            # Augment them
            self.augment_batch(x_batch, y_batch)

            yield x_batch, y_batch
