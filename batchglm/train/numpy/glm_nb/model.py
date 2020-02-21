import dask.array
import logging
import numpy as np
import scipy.sparse
import scipy.special
import sparse

from .external import Model, ModelIwls, InputDataGLM
from .processModel import ProcessModel

logger = logging.getLogger(__name__)


class ModelIwlsNb(ModelIwls, Model, ProcessModel):

    compute_mu: bool
    compute_r: bool

    def __init__(
            self,
            input_data: InputDataGLM,
            model_vars,
            compute_mu,
            compute_r,
            dtype,
    ):
        self.compute_mu = compute_mu
        self.compute_r = compute_r

        super(Model, self).__init__(
            input_data=input_data
        )
        ModelIwls.__init__(
            self=self,
            model_vars=model_vars
        )

    @property
    def fim_weight_aa(self):
        """

        :return: observations x features
        """
        return - self.location * self.scale / (self.scale + self.location)

    @property
    def ybar(self) -> np.ndarray:
        """

        :return: observations x features
        """
        return np.asarray(self.x - self.location) / self.location

    def fim_weight_aa_j(self, j):
        """

        :return: observations x features
        """
        return - self.location_j(j=j) * self.scale_j(j=j) / (self.scale_j(j=j) + self.location_j(j=j))

    def ybar_j(self, j) -> np.ndarray:
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
    def jac_weight_b(self):
        """

        :return: observations x features
        """
        scale = self.scale
        loc = self.location
        if isinstance(self.x, scipy.sparse.csr_matrix):
            scale_plus_x = np.asarray(scale + self.x)
        else:
            scale_plus_x = scale + self.x
        r_plus_mu = scale + loc

        # Define graphs for individual terms of constant term of hessian:
        const1 = scipy.special.digamma(scale_plus_x) - scipy.special.digamma(scale)
        const2 = - scale_plus_x / r_plus_mu
        const3 = np.log(scale) + np.ones_like(scale) - np.log(r_plus_mu)
        return scale * (const1 + const2 + const3)

    def jac_weight_b_j(self, j):
        """

        :return: observations x features
        """
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        scale = self.scale_j(j=j)
        loc = self.location_j(j=j)
        if isinstance(self.x, scipy.sparse.csr_matrix):
            scale_plus_x = np.asarray(scale + self.x[:, j])
        else:
            scale_plus_x = scale + self.x[:, j]
        r_plus_mu = scale + loc

        # Define graphs for individual terms of constant term of hessian:
        const1 = scipy.special.digamma(scale_plus_x) - scipy.special.digamma(scale)
        const2 = - scale_plus_x / r_plus_mu
        const3 = np.log(scale) + np.ones_like(scale) - np.log(r_plus_mu)
        return scale * (const1 + const2 + const3)

    @property
    def fim_ab(self) -> np.ndarray:
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
        return np.zeros([self.b_var.shape[1], 0, 0])

    @property
    def fim_bb(self) -> np.ndarray:
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
        return np.zeros([self.b_var.shape[1], 0, 0])

    @property
    def hessian_weight_ab(self):
        scale = self.scale
        loc = self.location
        return np.multiply(
            loc * scale,
            np.asarray(self.x - loc) / np.square(loc + scale)
        )

    @property
    def hessian_weight_aa(self):
        scale = self.scale
        loc = self.location
        if isinstance(self.x, np.ndarray) or isinstance(self.x, dask.array.core.Array):
            x_by_scale_plus_one = self.x / scale + np.ones_like(scale)
        else:
            x_by_scale_plus_one = np.asarray(self.x.divide(scale) + np.ones_like(scale))

        return - loc * x_by_scale_plus_one / np.square((loc / scale) + np.ones_like(loc))

    @property
    def hessian_weight_bb(self):
        scale = self.scale
        loc = self.location
        scale_plus_x = np.asarray(self.x + scale)
        scale_plus_loc = scale + loc
        # Define graphs for individual terms of constant term of hessian:
        const1 = scipy.special.digamma(scale_plus_x) + scale * scipy.special.polygamma(n=1, x=scale_plus_x)
        const2 = - scipy.special.digamma(scale) + scale * scipy.special.polygamma(n=1, x=scale)
        const3 = - loc * scale_plus_x + np.ones_like(scale) * 2. * scale * scale_plus_loc / np.square(scale_plus_loc)
        const4 = np.log(scale) + np.ones_like(scale) * 2. - np.log(scale_plus_loc)
        return scale * (const1 + const2 + const3 + const4)

    @property
    def ll(self):
        scale = self.scale
        loc = self.location
        log_r_plus_mu = np.log(scale + loc)
        if isinstance(self.x, np.ndarray) or isinstance(self.x, dask.array.core.Array):
            # dense numpy or dask
            ll = scipy.special.gammaln(scale + self.x) - \
                 scipy.special.gammaln(self.x + np.ones_like(scale)) - \
                 scipy.special.gammaln(scale) + \
                 self.x * (self.eta_loc - log_r_plus_mu) + \
                 np.multiply(scale, self.eta_scale - log_r_plus_mu)
        else:
            # sparse scipy
            ll = scipy.special.gammaln(np.asarray(scale + self.x)) - \
                scipy.special.gammaln(self.x + np.ones_like(scale)) - \
                scipy.special.gammaln(scale) + \
                np.asarray(self.x.multiply(self.eta_loc - log_r_plus_mu) +
                           np.multiply(scale, self.eta_scale - log_r_plus_mu))
            ll = np.asarray(ll)
        return self.np_clip_param(ll, "ll")

    def ll_j(self, j):
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        scale = self.scale_j(j=j)
        loc = self.location_j(j=j)
        log_r_plus_mu = np.log(scale + loc)
        if isinstance(self.x, np.ndarray) or isinstance(self.x, dask.array.core.Array):
            # dense numpy or dask
            ll = scipy.special.gammaln(scale + self.x[:, j]) - \
                 scipy.special.gammaln(self.x[:, j] + np.ones_like(scale)) - \
                 scipy.special.gammaln(scale) + \
                 self.x[:, j] * (self.eta_loc_j(j=j) - log_r_plus_mu) + \
                 np.multiply(scale, self.eta_scale_j(j=j) - log_r_plus_mu)
        else:
            # sparse scipy
            ll = scipy.special.gammaln(np.asarray(scale + self.x[:, j])) - \
                 scipy.special.gammaln(self.x + np.ones_like(scale)) - \
                 scipy.special.gammaln(scale) + \
                 np.asarray(self.x[:, j].multiply(self.eta_loc_j(j=j) - log_r_plus_mu) +
                            np.multiply(scale, self.eta_scale_j(j=j) - log_r_plus_mu))
            ll = np.asarray(ll)
        return self.np_clip_param(ll, "ll")

    def ll_handle(self):
        def fun(x, eta_loc, b_var, xh_scale):
            eta_scale = np.matmul(xh_scale, b_var)
            scale = np.exp(eta_scale)
            loc = np.exp(eta_loc)
            log_r_plus_mu = np.log(scale + loc)
            if isinstance(x, np.ndarray) or isinstance(x, dask.array.core.Array):
                # dense numpy or dask
                ll = scipy.special.gammaln(scale + x) - \
                     scipy.special.gammaln(x + np.ones_like(scale)) - \
                     scipy.special.gammaln(scale) + \
                     x * (eta_loc - log_r_plus_mu) + \
                     np.multiply(scale, eta_scale - log_r_plus_mu)
            else:
                raise ValueError("type x %s not supported" % type(x))
            return self.np_clip_param(ll, "ll")
        return fun

    def jac_b_handle(self):
        def fun(x, eta_loc, b_var, xh_scale):
            scale = np.exp(b_var)
            loc = np.exp(eta_loc)
            scale_plus_x = scale + x
            r_plus_mu = scale + loc

            # Define graphs for individual terms of constant term of hessian:
            const1 = scipy.special.digamma(scale_plus_x) - scipy.special.digamma(scale)
            const2 = - scale_plus_x / r_plus_mu
            const3 = np.log(scale) + np.ones_like(scale) - np.log(r_plus_mu)
            return scale * (const1 + const2 + const3)

        return fun
