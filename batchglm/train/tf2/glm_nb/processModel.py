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

        sf = dtype(pkg_constants.ACCURACY_MARGIN_RELATIVE_TO_LIMIT)
        bounds_min = {
            "a_var": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "b_var": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "eta_loc": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "eta_scale": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "loc": np.nextafter(0, np.inf, dtype=dtype),
            "scale": np.nextafter(0, np.inf, dtype=dtype),
            "probs": dtype(0),
            "log_probs": np.log(np.nextafter(0, np.inf, dtype=dtype)),
        }
        bounds_max = {
            "a_var": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "b_var": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "eta_loc": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "eta_scale": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "loc": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "scale": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "probs": dtype(1),
            "log_probs": dtype(0),
        }
        return bounds_min, bounds_max
