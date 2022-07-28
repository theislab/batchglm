import math
from typing import Union, Callable

import dask
import numpy as np

from .external import BaseModelContainer

def ll(scale, loc, x):
    resid = loc - x
    ll = -.5 * np.log(2 * math.pi) - np.log(scale) - .5 * np.power(resid / scale, 2)
    return ll

class ModelContainer(BaseModelContainer):

    @property
    def fim_weight_location_location(self) -> Union[np.ndarray, dask.array.core.Array]:
        return -self.location * self.scale / (self.scale + self.location)

    def fim_weight_location_location_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        return -self.location_j(j=j) * self.scale_j(j=j) / (self.scale_j(j=j) + self.location_j(j=j))

    @property
    def jac_weight_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        return -np.ones_like(self.x) - np.power((self.x - self.location) / self.scale, 2)

    def jac_weight_scale_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        return -np.ones_like(self.x[:, j]) - np.power((self.x[:, j] - self.location_j(j=j)) / self.scale_j(j=j), 2)

    @property
    def fim_location_location(self):
        return np.power(self.location / self.scale, 2)

    @property
    def fim_location_scale(self) -> np.ndarray:
        return np.zeros([self.theta_scale.shape[1], 0, 0])

    @property
    def fim_scale_scale(self) -> np.ndarray:
        return np.full([self.theta_scale.shape[1], 0, 0], 2)

    @property
    def hessian_weight_location_scale(self) -> np.ndarray:
        scale = self.scale
        loc = self.location
        return (-2 / np.power(scale, 3)) * loc * (self.x - loc)

    @property
    def hessian_weight_location_location(self) -> np.ndarray:
        scale = self.scale
        loc = self.location
        return (loc / np.power(scale, 2)) * (self.x - 2 * loc)

    @property
    def hessian_weight_scale_scale(self) -> np.ndarray:
        scale = self.scale
        loc = self.location
        return (-2 / np.power(scale, 2)) * np.power(self.x - loc, 2)

    @property
    def ll(self) -> Union[np.ndarray, dask.array.core.Array]:
        loc = self.location
        scale = self.scale
        x = self.model.x
        return np.asarray(ll(scale, loc, x))

    def ll_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]

        loc = self.location_j(j=j)
        scale = self.scale_j(j=j)
        resid = loc - self.model.x[:, j]
        ll = -.5 * np.log(2 * math.pi) - np.log(scale) - .5 * np.power(resid / scale, 2)
        return ll

    def ll_handle(self) -> Callable:
        def fun(x, eta_loc, theta_scale, xh_scale):
            eta_scale = np.matmul(xh_scale, theta_scale)
            scale = self.model.inverse_link_scale(eta_scale)
            loc = self.model.inverse_link_loc(eta_loc)
            return ll(scale, loc, x)

        return fun
