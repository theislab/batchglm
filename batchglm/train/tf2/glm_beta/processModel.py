from .external import ProcessModelGLM
import tensorflow as tf
import numpy as np
from .external import pkg_constants


class ProcessModel(ProcessModelGLM):

    def param_bounds(
            self,
            dtype
    ):
        if isinstance(dtype, tf.DType):
            dmax = dtype.max
            dtype = dtype.as_numpy_dtype
        else:
            dtype = np.dtype(dtype)
            dmax = np.finfo(dtype).max
            dtype = dtype.type

        zero = np.nextafter(0, np.inf, dtype=dtype)
        one = np.nextafter(1, -np.inf, dtype=dtype)

        sf = dtype(pkg_constants.ACCURACY_MARGIN_RELATIVE_TO_LIMIT)
        bounds_min = {
            "a_var": np.log(zero / (1 - zero)) / sf,
            "b_var": np.log(zero) / sf,
            "eta_loc": np.log(zero / (1 - zero)) / sf,
            "eta_scale": np.log(zero) / sf,
            "mean": np.nextafter(0, np.inf, dtype=dtype),
            "samplesize": np.nextafter(0, np.inf, dtype=dtype),
            "probs": dtype(0),
            "log_probs": np.log(zero),
        }
        bounds_max = {
            "a_var": np.log(one / (1 - one)) / sf,
            "b_var": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "eta_loc": np.log(one / (1 - one)) / sf,
            "eta_scale": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "mean": one,
            "samplesize": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "probs": dtype(1),
            "log_probs": dtype(0),
        }
        return bounds_min, bounds_max
