import abc
import numpy as np
from typing import Union
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

        - par_link_loc, par_link_scale: Model parameters in linker space.
        - location, scale: Model parameters in output space.
        - link_loc, link_scale: Transform output support to model parameter support.
        - inverse_link_loc: Transform model parameter support to output support.
        - design_loc, design_scale: design matrices
    """

    def __init__(
            self,
            input_data: InputDataGLM
    ):
        _ModelBase.__init__(
            self=self,
            input_data=input_data
        )
        self._a_var = None
        self._b_var = None

    @property
    def design_loc(self) -> np.ndarray:
        if self.input_data is None:
            return None
        else:
            return self.input_data.design_loc

    @property
    def design_scale(self) -> np.ndarray:
        if self.input_data is None:
            return None
        else:
            return self.input_data.design_scale

    @property
    def constraints_loc(self) -> np.ndarray:
        if self.input_data is None:
            return None
        else:
            return self.input_data.constraints_loc

    @property
    def constraints_scale(self) -> np.ndarray:
        if self.input_data is None:
            return None
        else:
            return self.input_data.constraints_scale

    @property
    def design_loc_names(self) -> list:
        if self.input_data is None:
            return None
        else:
            return self.input_data.design_loc_names

    @property
    def design_scale_names(self) -> list:
        if self.input_data is None:
            return None
        else:
            return self.input_data.design_scale_names

    @property
    def loc_names(self) -> list:
        if self.input_data is None:
            return None
        else:
            return self.input_data.loc_names

    @property
    def scale_names(self) -> list:
        if self.input_data is None:
            return None
        else:
            return self.input_data.scale_names

    @abc.abstractmethod
    def eta_loc(self) -> np.ndarray:
        pass

    @property
    def eta_scale(self) -> np.ndarray:
        eta = np.matmul(self.design_scale, self.b)
        eta = self.np_clip_param(eta, "eta_scale")
        return eta

    @property
    def location(self):
        return self.inverse_link_loc(self.eta_loc)

    @property
    def scale(self):
        return self.inverse_link_scale(self.eta_scale)

    @abc.abstractmethod
    def eta_loc_j(self, j) -> np.ndarray:
        pass

    def eta_scale_j(self, j) -> np.ndarray:
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        return np.matmul(self.design_scale, self.b[:, j])

    def location_j(self, j):
        return self.inverse_link_loc(self.eta_loc_j(j=j))

    def scale_j(self, j):
        return self.inverse_link_scale(self.eta_scale_j(j=j))

    @property
    def size_factors(self) -> Union[np.ndarray, None]:
        if self.input_data is None:
            return None
        else:
            return self.input_data.size_factors

    @property
    def a_var(self) -> np.ndarray:
        return self._a_var

    @property
    def b_var(self) -> np.ndarray:
        return self._b_var

    @property
    def a(self) -> np.ndarray:
        return np.dot(self.constraints_loc, self.a_var)

    @property
    def b(self) -> np.ndarray:
        return np.dot(self.constraints_scale, self.b_var)

    @abc.abstractmethod
    def link_loc(self, data):
        pass

    @abc.abstractmethod
    def link_scale(self, data):
        pass

    @abc.abstractmethod
    def inverse_link_loc(self, data):
        pass

    @abc.abstractmethod
    def inverse_link_scale(self, data):
        pass
