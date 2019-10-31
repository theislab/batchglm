import numpy as np

from .external import ProcessModelGlm
from .external import pkg_constants


class ProcessModel(ProcessModelGlm):

    def param_bounds(
            self,
            dtype
    ):
        dtype = np.dtype(dtype)
        dmin = np.finfo(dtype).min
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
            "likelihood": dtype(0),
            "ll": np.log(np.nextafter(0, np.inf, dtype=dtype)),
        }
        bounds_max = {
            "a_var": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "b_var": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "eta_loc": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "eta_scale": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "loc": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "scale": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "likelihood": dtype(1),
            "ll": dtype(0),
        }
        return bounds_min, bounds_max
