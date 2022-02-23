import abc
from typing import Any, Callable, Dict, Optional

try:
    import anndata
except ImportError:
    anndata = None
import numpy as np

from .external import _ModelGLM


class Model(_ModelGLM, metaclass=abc.ABCMeta):
    """
    Generalized Linear Model (GLM) with beta distributed noise, logit link for location and log link for scale.
    """

    def link_loc(self, data):
        return np.log(1 / (1 / data - 1))

    def inverse_link_loc(self, data):
        return 1 / (1 + np.exp(-data))

    def link_scale(self, data):
        return np.log(data)

    def inverse_link_scale(self, data):
        return np.exp(data)

    @property
    def eta_loc(self) -> np.ndarray:
        eta = np.matmul(self.design_loc, self.theta_location_constrained)
        assert self.size_factors is None, "size factors not allowed"
        return eta

    def eta_loc_j(self, j) -> np.ndarray:
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        eta = np.matmul(self.design_loc, self.theta_location_constrained[:, j])
        assert self.size_factors is None, "size factors not allowed"
        eta = self.np_clip_param(eta, "eta_loc")
        return eta

    # Re-parameterizations:

    @property
    def mean(self) -> np.ndarray:
        return self.location

    @property
    def samplesize(self) -> np.ndarray:
        return self.scale

    @property
    def p(self) -> np.ndarray:
        return self.mean * self.samplesize

    @property
    def q(self) -> np.ndarray:
        return (1 - self.mean) * self.samplesize

    # parameter contraints:

    def bounds(self, sf, dmax, dtype) -> Dict[str, Any]:

        zero = np.nextafter(0, np.inf, dtype=dtype)
        one = np.nextafter(1, -np.inf, dtype=dtype)

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

    # simulator:

    @property
    def rand_fn_ave(self) -> Optional[Callable]:
        return lambda shape: np.random.uniform(0.2, 0.8, shape)

    @property
    def rand_fn(self) -> Optional[Callable]:
        return None

    @property
    def rand_fn_loc(self) -> Optional[Callable]:
        return lambda shape: np.random.uniform(0.05, 0.15, shape)

    @property
    def rand_fn_scale(self) -> Optional[Callable]:
        return lambda shape: np.random.uniform(0.2, 0.5, shape)

    def generate_data(self):
        """
        Sample random data based on beta distribution and parameters.
        """
        return np.random.beta(a=self.p, b=self.q, size=None)
