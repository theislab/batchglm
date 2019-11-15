import abc
import logging
import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


class ModelBase(tf.keras.Model, metaclass=abc.ABCMeta):

    def __init__(self):
        super(ModelBase, self).__init__()

    @abc.abstractmethod
    def call(self, inputs, training=False, mask=None):
        pass


class LossBase(tf.keras.losses.Loss, metaclass=abc.ABCMeta):

    def __init__(self):
        super(LossBase, self).__init__()

    @abc.abstractmethod
    def call(self, y_true, y_pred):
        pass


class ProcessModelBase:

    @abc.abstractmethod
    def param_bounds(self, dtype):
        pass

    def tf_clip_param(
            self,
            param,
            name
    ):
        bounds_min, bounds_max = self.param_bounds(param.dtype)
        return tf.clip_by_value(
            param,
            bounds_min[name],
            bounds_max[name]
        )

    def np_clip_param(
            self,
            param,
            name
    ):
        bounds_min, bounds_max = self.param_bounds(param.dtype)
        return np.clip(
            param,
            bounds_min[name],
            bounds_max[name]
        )