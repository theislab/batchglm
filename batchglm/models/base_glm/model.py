import abc
from typing import Optional, Union

import dask.array
import numpy as np

try:
    import anndata
except ImportError:
    anndata = None

from .external import _ModelBase
from .input import InputDataGLM


class _ModelGLM(_ModelBase, metaclass=abc.ABCMeta):
    """
    Generalized Linear Model base class.

    Every GLM contains parameters for a location and a scale model
    in a parameter specific linker space and a design matrix for
    each location and scale model.
    """

    _a_var: np.ndarray = None
    _b_var: np.ndarray = None

    def __init__(self, input_data: Optional[InputDataGLM] = None):
        """
        Create a new _ModelGLM object.

        :param input_data: Input data for the model
        
        """
        _ModelBase.__init__(self=self, input_data=input_data)

    @property
    def design_loc(self) -> Union[np.ndarray, dask.array.core.Array]:
        """location design matrix"""
        if self.input_data is None:
            return None
        else:
            return self.input_data.design_loc

    @property
    def design_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        """scale design matrix"""
        if self.input_data is None:
            return None
        else:
            return self.input_data.design_scale

    @property
    def constraints_loc(self) -> Union[np.ndarray, dask.array.core.Array]:
        """constrainted location design matrix"""
        if self.input_data is None:
            return None
        else:
            return self.input_data.constraints_loc

    @property
    def constraints_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        """constrained scale design matrix"""
        if self.input_data is None:
            return None
        else:
            return self.input_data.constraints_scale

    @property
    def design_loc_names(self) -> list:
        """column names from location design matrix"""
        if self.input_data is None:
            return None
        else:
            return self.input_data.design_loc_names

    @property
    def design_scale_names(self) -> list:
        """column names from scale design matrix"""
        if self.input_data is None:
            return None
        else:
            return self.input_data.design_scale_names

    @property
    def loc_names(self) -> list:
        """column names from constratined location design matrix"""
        if self.input_data is None:
            return None
        else:
            return self.input_data.loc_names

    @property
    def scale_names(self) -> list:
        """column names from constrained scale design matrix"""
        if self.input_data is None:
            return None
        else:
            return self.input_data.scale_names

    @abc.abstractmethod
    def eta_loc(self) -> Union[np.ndarray, dask.array.core.Array]:
        pass

    @property
    def eta_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        """eta from scale model"""
        eta = np.matmul(self.design_scale, self.b)
        eta = self.np_clip_param(eta, "eta_scale")
        return eta

    @property
    def location(self):
        """the inverse link function applied to eta for the location model (i.e the fitted location)"""
        return self.inverse_link_loc(self.eta_loc)

    @property
    def scale(self):
        """the inverse link function applied to eta for the scale model (i.e the fitted location)"""
        return self.inverse_link_scale(self.eta_scale)

    @abc.abstractmethod
    def eta_loc_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        """
        Method to be implemented that allows fast access to a given observation's eta in the location model
        :param j: The index of the observation sought
        """
        pass

    def eta_scale_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        """"
        Allows fast access to a given observation's eta in the location model
        :param j: The index of the observation sought
        """
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        return np.matmul(self.design_scale, self.b[:, j])

    def location_j(self, j):
        """
        Allows fast access to a given observation's fitted location
        :param j: The index of the observation sought
        """
        return self.inverse_link_loc(self.eta_loc_j(j=j))

    def scale_j(self, j):
        """
        Allows fast access to a given observation's fitted scale
        :param j: The index of the observation sought
        """
        return self.inverse_link_scale(self.eta_scale_j(j=j))

    @property
    def size_factors(self) -> Union[np.ndarray, None]:
        """"Constant scale factors of the mean model in the linker space"""
        if self.input_data is None:
            return None
        else:
            return self.input_data.size_factors

    @property
    def a_var(self) -> np.ndarray:
        """"Fitted location model parameters"""
        return self._a_var

    @property
    def b_var(self) -> np.ndarray:
        """"Fitted scale model parameters"""
        return self._b_var

    @property
    def a(self) -> Union[np.ndarray, dask.array.core.Array]:
        """"dot product of location constraints with location parameter giving new constrained parameters"""
        return np.dot(self.constraints_loc, self.a_var)

    @property
    def b(self) -> Union[np.ndarray, dask.array.core.Array]:
         """"dot product of scale constraints with scale parameter giving new constrained parameters"""
        return np.dot(self.constraints_scale, self.b_var)

    @abc.abstractmethod
    def link_loc(self, data):
        """link function for location model"""
        pass

    @abc.abstractmethod
    def link_scale(self, data):
        """link function for scale model"""
        pass

    @abc.abstractmethod
    def inverse_link_loc(self, data):
        """inverse link function for location model"""
        pass

    @abc.abstractmethod
    def inverse_link_scale(self, data):
        """inverse link function for scale model"""
        pass
