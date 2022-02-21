import numpy as np

from .external import _SimulatorGLM, pkg_constants
from .model import Model


class Simulator(_SimulatorGLM, Model):
    """
    Simulator for Generalized Linear Models (GLMs) with beta distributed noise.
    Uses a logit-linker function for loc and a log-linker function for scale.
    """

    def __init__(self, num_observations=1000, num_features=100):
        _SimulatorGLM.__init__(self=self, num_observations=num_observations, num_features=num_features)

    def param_bounds(self, dtype):

        dtype = np.dtype(dtype)
        # dmin = np.finfo(dtype).min
        dmax = np.finfo(dtype).max
        dtype = dtype.type

        zero = np.nextafter(0, np.inf, dtype=dtype)
        one = np.nextafter(1, -np.inf, dtype=dtype)

        sf = dtype(pkg_constants.ACCURACY_MARGIN_RELATIVE_TO_LIMIT)
        bounds_min = {
            "theta_location": np.log(zero / (1 - zero)) / sf,
            "theta_scale": np.log(zero) / sf,
            "eta_loc": np.log(zero / (1 - zero)) / sf,
            "eta_scale": np.log(zero) / sf,
            "mean": np.nextafter(0, np.inf, dtype=dtype),
            "samplesize": np.nextafter(0, np.inf, dtype=dtype),
            "probs": dtype(0),
            "log_probs": np.log(zero),
        }
        bounds_max = {
            "theta_location": np.log(one / (1 - one)) / sf,
            "theta_scale": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "eta_loc": np.log(one / (1 - one)) / sf,
            "eta_scale": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "mean": one,
            "samplesize": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "probs": dtype(1),
            "log_probs": dtype(0),
        }
        return bounds_min, bounds_max

    def generate_params(
        self,
        rand_fn_ave=lambda shape: np.random.uniform(0.2, 0.8, shape),
        rand_fn=None,
        rand_fn_loc=lambda shape: np.random.uniform(0.05, 0.15, shape),
        rand_fn_scale=lambda shape: np.random.uniform(0.2, 0.5, shape),
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
        Sample random data based on beta distribution and parameters.
        """
        data_matrix = np.random.beta(a=self.p, b=self.q, size=None)
        self.assemble_input_data(data_matrix, sparse)
