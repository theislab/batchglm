import abc
try:
    import anndata
except ImportError:
    anndata = None
import xarray as xr
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
    def eta_loc(self) -> xr.DataArray:
        eta = np.matmul(self.design_loc, self.a)
        if self.size_factors is not None:
            eta += np.expand_dims(self.size_factors, axis=1)
        return eta

    # Re-parameterizations:

    @property
    def mu(self) -> xr.DataArray:
        return self.location

    @property
    def phi(self) -> xr.DataArray:
        return self.scale
