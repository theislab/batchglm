import math
from typing import Union

import dask
import numpy as np

from .external import BaseModelContainer


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
        # Unfinished in manuscript?
        pass

    @property
    def fim_scale_scale(self) -> np.ndarray:
        # Unfinished in manuscript?
        pass

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
        resid = loc - self.model.x
        sd = np.sqrt(np.sum(np.power(resid, 2), 0))
        var = np.power(sd, 2)
        ll = -.5 * loc.shape[0] * np.log(2 * math.pi * var) - .5 * np.linalg.norm(resid, axis=0) / np.power(sd, 2)
        return ll

    def ll_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]

        loc = self.location_j(j=j)
        resid = loc - self.model.x[:, j]
        sd = np.sqrt(np.sum(np.power(resid, 2), 0))
        var = np.power(sd, 2)
        ll = -.5 * loc.shape[0] * np.log(2 * math.pi * var) - .5 * np.linalg.norm(resid, axis=0) / np.power(sd, 2)
        return ll
