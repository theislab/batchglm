import abc
import logging

import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


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
            bounds_max[name],
            # out=param
        )
