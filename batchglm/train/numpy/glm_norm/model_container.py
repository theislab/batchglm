import math
from typing import Union, Callable

import numpy as np
import dask

from .external import NumpyModelContainer
from .utils import ll
from ....utils.data import dask_compute


class ModelContainer(NumpyModelContainer):

    @property
    def fim_weight(self):
        raise NotImplementedError("This method is currently unimplemented as it isn't used by any built-in procedures.")

    @property
    def jac_weight(self):
        raise NotImplementedError("This method is currently unimplemented as it isn't used by any built-in procedures.")

    @property
    def jac_weight_j(self):
        raise NotImplementedError("This method is currently unimplemented as it isn't used by any built-in procedures.")

    @property
    def jac_location(self) -> Union[np.ndarray, dask.array.core.Array]:
        xh = self.xh_loc
        loc = self.location
        scale = self.scale
        x = self.x
        w = (loc - x) / np.power(scale, 2)
        return np.matmul(w.transpose(), xh)

    def jac_location_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        xh = self.xh_loc
        loc = self.location_j(j=j)
        scale = self.scale_j(j=j)
        x = self.x[:, j]
        w = (loc - x) / np.power(scale, 2)
        return np.matmul(w.transpose(), xh)

    @property
    def jac_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        """

        :return: (features x inferred param)
        """
        w = self.jac_weight_scale  # (observations x features)
        xh = self.xh_scale  # (observations x inferred param)
        return np.matmul(w.transpose(), xh)


    @dask_compute
    def jac_scale_j(self, j) -> np.ndarray:
        """

        :return: (features x inferred param)
        """
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        w = self.jac_weight_scale_j(j=j)  # (observations x features)
        xh = self.xh_scale  # (observations x inferred param)
        return np.matmul(w.transpose(), xh)

    @property
    def fim_weight_location_location(self) -> Union[np.ndarray, dask.array.core.Array]:
        return 1 / np.power(self.scale, 2)

    def fim_weight_location_location_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        return 1 / (self.scale_j(j=j) * self.scale_j(j=j))

    @property
    def jac_weight_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        return -np.ones_like(self.x) - np.power((self.x - self.location) / self.scale, 2)

    def jac_weight_scale_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        return -np.ones_like(self.x[:, j]) - np.power((self.x[:, j] - self.location_j(j=j)) / self.scale_j(j=j), 2)

    @property
    def fim_location_scale(self) -> np.ndarray:
        return np.zeros([self.model.x.shape[1], self.theta_location.shape[0], self.theta_scale.shape[0]])

    @property
    def fim_weight_scale_scale(self) -> np.ndarray:
        return np.full(self.scale.shape, 2)

    @property
    def fim_scale_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        """

        :return: (features x inferred param x inferred param)
        """
        w = self.fim_weight_scale_scale
        xh = self.xh_scale
        return np.einsum("fob,oc->fbc", np.einsum("ob,of->fob", xh, w), xh)

    @property
    def hessian_weight_location_scale(self) -> np.ndarray:
        scale = self.scale
        loc = self.location
        return (2 / np.power(scale, 2)) * (self.x - loc)

    @property
    def hessian_weight_location_location(self) -> np.ndarray:
        scale = self.scale
        return -1 / np.power(scale, 2)

    @property
    def hessian_weight_scale_scale(self) -> np.ndarray:
        scale = self.scale
        loc = self.location
        return (2 / np.power(scale, 2)) * np.power(self.x - loc, 2)

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
