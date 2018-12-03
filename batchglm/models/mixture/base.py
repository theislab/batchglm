import abc

import numpy as np
import xarray as xr

from ..base import BasicModel


class Model(BasicModel, metaclass=abc.ABCMeta):
    """
    Mixture Model base class.

    Every mixture model has the parameter `mixture_prob` which denotes the assignment probabilities of
    the samples to each mixture.
    They are linked to `location` and `scale` by a linker function, e.g. `location = exp(design_loc * link_loc)`.
    """

    def mixture_assignment(self, flat=True) -> xr.DataArray:
        if flat:
            retval: xr.DataArray = np.squeeze(np.argmax(self.mixture_prob(), axis=-1))
            return retval
        else:
            mixture_prob = self.mixture_prob()
            retval = mixture_prob.copy()
            retval.values = np.zeros_like(mixture_prob)
            retval[np.arange(np.shape(mixture_prob)[0]), np.argmax(mixture_prob, axis=-1)] = 1
            return retval

    @property
    def mixture_weights(self) -> xr.DataArray:
        return np.exp(self.mixture_log_weights)

    @property
    @abc.abstractmethod
    def mixture_log_weights(self) -> xr.DataArray:
        pass

    def mixture_prob(self) -> xr.DataArray:
        retval: xr.DataArray = np.exp(self.mixture_log_prob())
        return retval

    @abc.abstractmethod
    def mixture_log_prob(self) -> xr.DataArray:
        pass

    @property
    @abc.abstractmethod
    def mixture_weight_constraints(self) -> xr.DataArray:
        pass
