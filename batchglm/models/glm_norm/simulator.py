import numpy as np

from .model import Model
from .external import _SimulatorGLM
from .external import pkg_constants


class Simulator(_SimulatorGLM, Model):
    """
    Simulator for Generalized Linear Models (GLMs) with normal noise.
    Uses the identity as linker function for loc and a log-linker function for scale.
    """

    def __init__(
            self,
            num_observations=1000,
            num_features=100
    ):
        _SimulatorGLM.__init__(
            self=self,
            model=None,
            num_observations=num_observations,
            num_features=num_features
        )

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
            "a_var": np.nextafter(-dmax, np.inf, dtype=dtype) / sf,
            "b_var": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "eta_loc": np.nextafter(-dmax, np.inf, dtype=dtype) / sf,
            "eta_scale": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "mean": np.nextafter(-dmax, np.inf, dtype=dtype) / sf,
            "sd": np.nextafter(0, np.inf, dtype=dtype),
            "probs": dtype(0),
            "log_probs": np.log(np.nextafter(0, np.inf, dtype=dtype)),
        }
        bounds_max = {
            "a_var": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "b_var": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "eta_loc": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "eta_scale": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "mean": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "sd": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "probs": dtype(1),
            "log_probs": dtype(0),
        }
        return bounds_min, bounds_max

    def generate_params(
            self,
            rand_fn_ave=lambda shape: np.random.uniform(10, 1000, shape),
            rand_fn=None,
            rand_fn_loc=lambda shape: np.random.uniform(50, 100, shape),
            rand_fn_scale=lambda shape: np.random.uniform(1.5, 10, shape),
        ):
        self._generate_params(
            self,
            rand_fn_ave=rand_fn_ave,
            rand_fn=rand_fn,
            rand_fn_loc=rand_fn_loc,
            rand_fn_scale=rand_fn_scale,
        )

    def generate_data(self, sparse: bool = False):
        """
        Sample random data based on normal distribution and parameters.
        """
        data_matrix = np.random.normal(
            loc=self.mean,
            scale=self.sd,
            size=None
        )
        self.assemble_input_data(data_matrix, sparse)
