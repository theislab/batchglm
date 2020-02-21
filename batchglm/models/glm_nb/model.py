import abc
try:
    import anndata
except ImportError:
    anndata = None
import numpy as np

from .external import _ModelGLM


class Model(_ModelGLM, metaclass=abc.ABCMeta):
    """
    Generalized Linear Model (GLM) with negative binomial noise.
    """

    def link_loc(self, data):
        return np.log(data)

    def inverse_link_loc(self, data):
        return np.exp(data)

    def link_scale(self, data):
        return np.log(data)

    def inverse_link_scale(self, data):
        return np.exp(data)

    @property
    def eta_loc(self) -> np.ndarray:
        eta = np.matmul(self.design_loc, self.a)
        if self.size_factors is not None:
            eta += self.size_factors
        eta = self.np_clip_param(eta, "eta_loc")
        return eta

    def eta_loc_j(self, j) -> np.ndarray:
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        eta = np.matmul(self.design_loc, self.a[:, j])
        if self.size_factors is not None:
            eta += self.size_factors
        eta = self.np_clip_param(eta, "eta_loc")
        return eta

    # Re-parameterizations:

    @property
    def mu(self) -> np.ndarray:
        return self.location

    @property
    def phi(self) -> np.ndarray:
        return self.scale
