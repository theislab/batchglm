import math
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
        return -self.location * self.scale / (self.scale + self.location)

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
        return -self.location_j(j=j) * self.scale_j(j=j) / (self.scale_j(j=j) + self.location_j(j=j))

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
        """
        Scale model jacobian
        :return: observations x features
        """
        return -np.ones_like(self.x) - np.power((self.x - self.location) / self.scale, 2)

    def jac_weight_scale_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        """
        Scale model jacobian at location j
        :param j: Location
        :return: observations x features
        """
        return -np.ones_like(self.x[:, j]) - np.power((self.x[:, j] - self.location_j(j=j)) / self.scale_j(j=j), 2)

    @property
    def fim_location_location(self):
        return np.power(self.location / self.scale, 2)
        

    @property
    def fim_location_scale(self) -> np.ndarray:
        """
        Location-scale coefficient block of FIM

        The negative binomial model is not fit as whole with IRLS but only the location model.
        The location model is conditioned on the scale model estimates, which is why we only
        supply the FIM of the location model and return an empty FIM for scale model components.
        Note that there is also no closed form FIM for the scale-scale block. Returning a zero-array
        here leads to singular matrices for the whole location-scale FIM in some cases that throw
        linear algebra errors when inverted.

        :return: (features x inferred param x inferred param)
        """
        return np.zeros([self.theta_scale.shape[1], 0, 0])

    @property
    def fim_scale_scale(self) -> np.ndarray:
        """
        Scale-scale coefficient block of FIM

        The negative binomial model is not fit as whole with IRLS but only the location model.
        The location model is conditioned on the scale model estimates, which is why we only
        supply the FIM of the location model and return an empty FIM for scale model components.
        Note that there is also no closed form FIM for the scale-scale block. Returning a zero-array
        here leads to singular matrices for the whole location-scale FIM in some cases that throw
        linear algebra errors when inverted.

        :return: (features x inferred param x inferred param)
        """
        return np.zeros([self.theta_scale.shape[1], 0, 0])

    @property
    def hessian_weight_location_scale(self) -> np.ndarray:
        """scale-location block of the hessian matrix"""
        scale = self.scale
        loc = self.location
        return (-2 / np.power(scale, 3)) * loc * (self.x - loc)

    @property
    def hessian_weight_location_location(self) -> np.ndarray:
        """location-location block of the hessian matrix"""
        scale = self.scale
        loc = self.location
        return (loc / np.power(scale, 2)) * (self.x - 2 * loc)

    @property
    def hessian_weight_scale_scale(self) -> np.ndarray:
        """scale-scale block of the hessian matrix"""
        scale = self.scale
        loc = self.location
        return (-2 / np.power(scale, 2)) * np.power(self.x - loc, 2)

    @property
    def ll(self) -> Union[np.ndarray, dask.array.core.Array]:
        """log-likelihood"""
        scale = self.scale
        loc = self.location
        log_scale = np.log(scale)
        ll = -.5 * np.repeat(2 * math.pi, loc.shape[0])  - log_scale - np.power((self.x - loc) / (2 * scale), 2)
        return self.np_clip_param(ll, "ll")

    def ll_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        """
        Log likelhiood for observation j
        :param j: observation
        """
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]

        scale = self.scale_j(j=j)
        loc = self.location_j(j=j)
        log_scale = np.log(scale)
        ll = -.5 * np.repeat(2 * math.pi, loc.shape[0])  - log_scale - np.power((self.x - loc) / (2 * scale), 2)
        return self.np_clip_param(ll, "ll")

    def ll_handle(self) -> Callable: # what?
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

    def jac_scale_handle(self) -> Callable: # what?
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
