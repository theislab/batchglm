import abc
from typing import Any, Callable, Dict, Optional

try:
    import anndata
except ImportError:
    anndata = None
import numpy as np

from .external import _ModelGLM


class Model(_ModelGLM, metaclass=abc.ABCMeta):

    """Generalized Linear Model (GLM) with normal noise."""

    def link_loc(self, data):
        """Short summary.

        :param type data: Description of parameter `data`.
        :return: Description of returned object.
        :rtype: type

        """
        return data

    def inverse_link_loc(self, data):
        """Short summary.

        :param type data: Description of parameter `data`.
        :return: Description of returned object.
        :rtype: type

        """
        return data

    def link_scale(self, data):
        """Short summary.

        :param type data: Description of parameter `data`.
        :return: Description of returned object.
        :rtype: type

        """
        return np.log(data)

    def inverse_link_scale(self, data):
        """Short summary.

        :param type data: Description of parameter `data`.
        :return: Description of returned object.
        :rtype: type

        """
        return np.exp(data)

    @property
    def eta_loc(self) -> np.ndarray:
        """Short summary.

        :return: Description of returned object.
        :rtype: np.ndarray

        """
        eta = np.matmul(self.design_loc, self.theta_location_constrained)
        if self.size_factors is not None:
            eta *= np.expand_dims(self.size_factors, axis=1)
        return eta

    def eta_loc_j(self, j) -> np.ndarray:
        """Short summary.

        :param type j: Description of parameter `j`.
        :return: Description of returned object.
        :rtype: np.ndarray

        """
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        eta = np.matmul(self.design_loc, self.theta_location_constrained[:, j])
        if self.size_factors is not None:
            eta *= np.expand_dims(self.size_factors, axis=1)
        eta = self.np_clip_param(eta, "eta_loc")
        return eta

    # Re-parameterizations:

    @property
    def mean(self) -> np.ndarray:
        """Short summary.

        :return: Description of returned object.
        :rtype: np.ndarray

        """
        return self.location

    @property
    def sd(self) -> np.ndarray:
        """Short summary.

        :return: Description of returned object.
        :rtype: np.ndarray

        """
        return self.scale

    # param constraints:

    def bounds(self, sf, dmax, dtype) -> Dict[str, Any]:

        bounds_min = {
            "theta_location": np.nextafter(-dmax, np.inf, dtype=dtype) / sf,
            "theta_scale": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "eta_loc": np.nextafter(-dmax, np.inf, dtype=dtype) / sf,
            "eta_scale": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "mean": np.nextafter(-dmax, np.inf, dtype=dtype) / sf,
            "sd": np.nextafter(0, np.inf, dtype=dtype),
            "probs": dtype(0),
            "log_probs": np.log(np.nextafter(0, np.inf, dtype=dtype)),
        }
        bounds_max = {
            "theta_location": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "theta_scale": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "eta_loc": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "eta_scale": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "mean": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "sd": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "probs": dtype(1),
            "log_probs": dtype(0),
        }
        return bounds_min, bounds_max

    # simulator:

    @property
    def rand_fn_ave(self) -> Optional[Callable]:
        return lambda shape: np.random.uniform(10, 1000, shape)

    @property
    def rand_fn(self) -> Optional[Callable]:
        return None

    @property
    def rand_fn_loc(self) -> Optional[Callable]:
        return lambda shape: np.random.uniform(50, 100, shape)

    @property
    def rand_fn_scale(self) -> Optional[Callable]:
        return lambda shape: np.random.uniform(1.5, 10, shape)

    def generate_data(self):
        """
        Sample random data based on normal distribution and parameters.
        """
        return np.random.normal(loc=self.mean, scale=self.sd, size=None)
