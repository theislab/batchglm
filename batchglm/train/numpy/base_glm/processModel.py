import abc
import numpy as np


class ProcessModelGlm:

    @abc.abstractmethod
    def param_bounds(self, dtype):
        pass

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

