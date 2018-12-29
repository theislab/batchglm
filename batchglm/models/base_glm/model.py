import abc
from typing import Union

import xarray as xr

from .input import _InputData_GLM, INPUT_DATA_PARAMS
from .external import _Model_Base, _Model_XArray_Base

# Define distribution parameters:
MODEL_PARAMS = INPUT_DATA_PARAMS.copy()
MODEL_PARAMS.update({
    "sigma2": ("observations", "features"),
    "probs": ("observations", "features"),
    "log_probs": ("observations", "features"),
    "log_likelihood": (),
    "a": ("design_loc_params", "features"),
    "b": ("design_scale_params", "features"),
    "par_link_loc": ("design_loc_params", "features"),
    "par_link_scale": ("design_scale_params", "features"),
})

class _Model_GLM(_Model_Base, metaclass=abc.ABCMeta):
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

    @property
    def design_loc(self) -> xr.DataArray:
        return self.input_data.design_loc

    @property
    def design_scale(self) -> xr.DataArray:
        return self.input_data.design_scale

    @property
    @abc.abstractmethod
    def a(self) -> xr.DataArray:
        pass

    @property
    @abc.abstractmethod
    def b(self) -> xr.DataArray:
        pass

    @property
    def par_link_loc(self):
        return self.a

    @property
    def par_link_scale(self):
        return self.b

    @abc.abstractmethod
    def location(self):
        pass

    @abc.abstractmethod
    def scale(self):
        pass

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

    @property
    def size_factors(self) -> Union[xr.DataArray, None]:
        return self.input_data.size_factors


class _Model_XArray_GLM(_Model_XArray_Base):
    _input_data: _InputData_GLM
    params: xr.Dataset

    def __init__(self, input_data: _InputData_GLM, params: xr.Dataset):
        super(_Model_XArray_Base, self).__init__(input_data=input_data, params=params)

    @property
    def a(self):
        return self.params['a']

    @property
    def b(self):
        return self.params['b']

    def __str__(self):
        return "[%s.%s object at %s]: data=%s" % (
            type(self).__module__,
            type(self).__name__,
            hex(id(self)),
            self.params
        )

    def __repr__(self):
        return self.__str__()
