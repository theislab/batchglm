from typing import Callable, Union

import dask
import numpy as np
import scipy

from .external import BaseModelContainer


class ModelContainer(BaseModelContainer):
    @property
    def fim_weight(self):
        raise NotImplementedError("This method is currently unimplemented as it isn't used by any built-in procedures.")

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
    def jac_weight(self):
        raise NotImplementedError("This method is currently unimplemented as it isn't used by any built-in procedures.")

    @property
    def jac_weight_j(self):
        raise NotImplementedError("This method is currently unimplemented as it isn't used by any built-in procedures.")

    @property
    def jac_weight_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        pass

    def jac_weight_scale_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        pass

    @property
    def fim_location_scale(self) -> np.ndarray:
        pass

    @property
    def fim_scale_scale(self) -> np.ndarray:
        pass

    @property
    def hessian_weight_location_scale(self) -> np.ndarray:
        pass

    @property
    def hessian_weight_location_location(self) -> np.ndarray:
        """location-location block of the hessian matrix"""
        return self.location

    @property
    def hessian_weight_scale_scale(self) -> np.ndarray:
        pass

    @property
    def ll(self) -> Union[np.ndarray, dask.array.core.Array]:
        """log-likelihood"""
        loc = self.location
        log_loc = np.log(loc)
        x_times_log_loc = self.x * log_loc
        log_x_factorial = np.log(scipy.special.gammaln(self.x + np.ones_like(self.x)))
        ll = x_times_log_loc - log_loc - log_x_factorial
        return self.np_clip_param(ll, "ll")

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
        return self.np_clip_param(ll, "ll")

    def ll_handle(self) -> Callable: # what does this do?
        def fun(x, eta_loc, theta_scale, xh_scale):
            eta_scale = np.matmul(xh_scale, theta_scale)
            scale = np.exp(eta_scale)
            loc = np.exp(eta_loc)
            log_r_plus_mu = np.log(scale + loc)
            if isinstance(x, np.ndarray) or isinstance(x, dask.array.core.Array):
                # dense numpy or dask
                ll = (
                    scipy.special.gammaln(scale + x)
                    - scipy.special.gammaln(x + np.ones_like(scale))
                    - scipy.special.gammaln(scale)
                    + x * (eta_loc - log_r_plus_mu)
                    + np.multiply(scale, eta_scale - log_r_plus_mu)
                )
            else:
                raise ValueError("type x %s not supported" % type(x))
            return self.np_clip_param(ll, "ll")

        return fun

    def jac_scale_handle(self) -> Callable:
        def fun(x, eta_loc, theta_scale, xh_scale):
            scale = np.exp(theta_scale)
            loc = np.exp(eta_loc)
            scale_plus_x = scale + x
            r_plus_mu = scale + loc

            # Define graphs for individual terms of constant term of hessian:
            const1 = scipy.special.digamma(scale_plus_x) - scipy.special.digamma(scale)
            const2 = -scale_plus_x / r_plus_mu
            const3 = np.log(scale) + np.ones_like(scale) - np.log(r_plus_mu)
            return scale * (const1 + const2 + const3)

        return fun
