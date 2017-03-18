# This is a callback function to be used with training of Keras models.
# It create an exponential moving average of a model (trainable) weights.
# This functionlity is already available in TensorFlow:
# https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html#ExponentialMovingAverage
# and can often be used to get better validation/test performance. For an
# intuitive explantion on why to use this, see 'Model Ensembles" section here:
# http://cs231n.github.io/neural-networks-3/

import numpy as np

from keras import backend as K
from keras.callbacks import Callback

import warnings


class ExponentialMovingAverage(Callback):
    """Create a copy of trainable weights which gets updated at every
       batch using exponential weight decay. The moving average weights along
       with the other states of original model(except original model trainable
       weights) will be saved at every epoch if save_ema_model is True.
       If both save_ema_model and save_best_only are True, the latest
       best moving average model according to the quantity monitored
       will not be overwritten. Of course, save_best_only can be True
       only if there is a validation set.
       This is equivalent to save_best_only mode of ModelCheckpoint
       callback with similar code. custom_objects is a dictionary
       holding name and Class implementation for custom layers.
       At end of every batch, the update is as follows:
       mv_weight -= (1 - decay) * (mv_weight - weight)
       where weight and mv_weight is the ordinal model weight and the moving
       averaged weight respectively. At the end of each epoch, the moving
       averaged weights are transferred to the original model (so it may be
       used in later callbacks) and at the epoch beginning original weights
       are restored.
       """
    def __init__(self, decay=0.999, filepath='temp_weight.hdf5',
                 save_ema_model=True, verbose=0,
                 save_best_only=False, monitor='val_loss', mode='auto'):
        self.decay = decay
        self.filepath = filepath
        self.verbose = verbose
        self.save_ema_model = save_ema_model
        self.save_best_only = save_best_only
        self.monitor = monitor

        super(ExponentialMovingAverage, self).__init__()

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_begin(self, logs={}):
        # Initialize moving averaged weights using original model values
        self.cur_trainable_weights_vals = {x.name: K.get_value(x) for x in self.model.trainable_weights}
        self.ema_trainable_weights_vals = {x.name: K.get_value(x) for x in self.model.trainable_weights}
        pass

    def on_batch_end(self, batch, logs={}):
        for w in self.model.trainable_weights:
            old_val = self.ema_trainable_weights_vals[w.name]
            new_val = K.get_value(w)

            self.ema_trainable_weights_vals[w.name] -= (1.0 - self.decay) * (old_val - new_val)

    def on_epoch_begin(self, epoch, logs={}):
        """When starting each epoch, we restore model weights to their current values"""

        for w in self.model.trainable_weights:
            K.set_value(w, self.cur_trainable_weights_vals[w.name])

    def on_epoch_end(self, epoch, logs={}):
        """After each epoch, we transfer ema weights to the model and optionally save it"""

        # Save current weights and replace them with ema
        for w in self.model.trainable_weights:
            self.cur_trainable_weights_vals[w.name] = K.get_value(w)
            K.set_value(w, self.ema_trainable_weights_vals[w.name])

        if self.save_ema_model:
            filepath = self.filepath.format(epoch=epoch, **logs)

            if self.save_best_only:
                current = logs.get(self.monitor)

                if current is None:
                    warnings.warn('Can save best moving averaged model only '
                                  'with %s available, skipping.'
                                  % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('saving moving average model to %s'
                                  % (filepath))
                        self.best = current
                        self.model.save_weights(filepath, overwrite=True)
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving moving average model to %s' % (epoch, filepath))

                self.model.save_weights(filepath, overwrite=True)
