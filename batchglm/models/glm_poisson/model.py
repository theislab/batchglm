import abc
from typing import Any, Callable, Dict, Optional, Tuple, Union

import dask.array
import numpy as np

from .external import _ModelGLM


class Model(_ModelGLM, metaclass=abc.ABCMeta):
    """
    Generalized Linear Model (GLM) with Poisson noise.
    """

    def link_loc(self, data) -> Union[np.ndarray, dask.array.core.Array]:
        return np.log(data)

    def inverse_link_loc(self, data) -> Union[np.ndarray, dask.array.core.Array]:
        return np.exp(data)

    def link_scale(self, data) -> Union[np.ndarray, dask.array.core.Array]:
        return np.log(data)

    def inverse_link_scale(self, data) -> Union[np.ndarray, dask.array.core.Array]:
        return np.exp(data)

    @property
    def eta_loc(self) -> Union[np.ndarray, dask.array.core.Array]:
        eta = np.matmul(self.design_loc, self.theta_location_constrained)
        if self.size_factors is not None:
            eta += self.size_factors
        eta = self.np_clip_param(eta, "eta_loc")
        return eta

    def eta_loc_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        eta = np.matmul(self.design_loc, self.theta_location_constrained[:, j])
        if self.size_factors is not None:
            eta += self.size_factors
        eta = self.np_clip_param(eta, "eta_loc")
        return eta

    # Re-parameterizations:

    @property
    def lam(self) -> Union[np.ndarray, dask.array.core.Array]:
        return self.location

    # param constraints:

    def bounds(self, sf, dmax, dtype) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        bounds_min = {
            "theta_location": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "theta_scale": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "eta_loc": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "eta_scale": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "loc": np.nextafter(0, np.inf, dtype=dtype),
            "scale": np.nextafter(0, np.inf, dtype=dtype),
            "likelihood": dtype(0),
            "ll": np.log(np.nextafter(0, np.inf, dtype=dtype)),
        }
        bounds_max = {
            "theta_location": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "theta_scale": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "eta_loc": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "eta_scale": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "loc": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "scale": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "likelihood": dtype(1),
            "ll": dtype(0),
        }
        return bounds_min, bounds_max

    # simulator:

    @property
    def rand_fn_ave(self) -> Optional[Callable]:
        return lambda shape: np.random.poisson(500, shape) + 1

    @property
    def rand_fn(self) -> Optional[Callable]:
        return lambda shape: np.abs(np.random.uniform(0.5, 2, shape))

    @property
    def rand_fn_loc(self) -> Optional[Callable]:
        return None

    @property
    def rand_fn_scale(self) -> Optional[Callable]:
        return None

    def generate_data(self) -> np.ndarray:
        """
        Sample random data based on negative binomial distribution and parameters.
        """
        return np.random.poisson(lam=self.lam)
