from typing import Callable, Union

import dask
import numpy as np
import scipy

from .external import BaseModelContainer

class ModelContainer(BaseModelContainer):
    @property
    def fim_weight_location_location(self) -> Union[np.ndarray, dask.array.core.Array]:
        """
        Fisher inverse matrix weights
        :return: observations x features
        """
        return self.location

    @property
    def ybar(self) -> Union[np.ndarray, dask.array.core.Array]:
        """
        :return: observations x features
        """
        return np.asarray(self.x - self.location) / self.location

    def fim_weight_location_location_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        """
        Fisher inverse matrix weights at j
        :return: observations x features
        """
        return self.location_j(j=j)

    def ybar_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        """
        :return: observations x features
        """
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        if isinstance(self.x, np.ndarray) or isinstance(self.x, dask.array.core.Array):
            return (self.x[:, j] - self.location_j(j=j)) / self.location_j(j=j)
        else:
            return np.asarray(self.x[:, j] - self.location_j(j=j)) / self.location_j(j=j)

    @property
    def hessian_weight_location_location(self) -> np.ndarray:
        """location-location block of the hessian matrix"""
        return self.location

    @property
    def ll(self) -> Union[np.ndarray, dask.array.core.Array]:
        """log-likelihood"""
        loc = self.location
        log_loc = np.log(loc)
        x_times_log_loc = self.x * log_loc
        log_x_factorial = np.log(scipy.special.gammaln(self.x + np.ones_like(self.x)))
        ll = x_times_log_loc - log_loc - log_x_factorial
        return np.asarray(self.np_clip_param(ll, "ll"))

    def ll_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        """
        Log likelhiood for observation j
        :param j: observation
        """
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        log_loc = np.log(self.location_j(j=j))
        x_times_log_loc = self.x[:, j] * log_loc
        log_x_factorial = np.log(scipy.special.gammaln(self.x[:, j] + np.ones_like(self.x[:, j])))
        ll = x_times_log_loc - log_loc - log_x_factorial
        return np.asarray(self.np_clip_param(ll, "ll"))

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
    def fim_location_scale(self) -> np.ndarray:
        raise NotImplementedError("This method is currently unimplemented as it isn't used by any built-in procedures.")
