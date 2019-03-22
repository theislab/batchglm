import abc
from typing import Union
try:
    import anndata
except ImportError:
    anndata = None

import numpy as np
import xarray as xr

from .input import InputData, INPUT_DATA_PARAMS
from .external import _Model_Base, _Model_XArray_Base

# Define distribution parameters:
MODEL_PARAMS = INPUT_DATA_PARAMS.copy()
MODEL_PARAMS.update({
    "sigma2": ("observations", "features"),
    "probs": ("observations", "features"),
    "log_probs": ("observations", "features"),
    "log_likelihood": (),
    "a_var": ("loc_params", "features"),
    "b_var": ("scale_params", "features"),
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
    def constraints_loc(self) -> xr.DataArray:
        return self.input_data.constraints_loc

    @property
    def constraints_scale(self) -> xr.DataArray:
        return self.input_data.constraints_scale

    @property
    def par_link_loc(self):
        return self.a

    @property
    def par_link_scale(self):
        return self.b

    @abc.abstractmethod
    def eta_loc(self) -> xr.DataArray:
        pass

    @property
    def eta_scale(self) -> xr.DataArray:
        # TODO: take this switch out once xr.dataset slicing yields dataarray with loc_names coordinate:
        if isinstance(self.par_link_loc, xr.DataArray):
            eta = self.design_scale.dot(self.par_link_scale, dims="design_scale_params")
        else:
            eta = np.matmul(self.design_scale.values, self.par_link_scale)

        return eta

    @property
    def location(self):
        return self.inverse_link_loc(self.eta_loc)

    @property
    def scale(self):
        return self.inverse_link_scale(self.eta_scale)

    @property
    def size_factors(self) -> Union[xr.DataArray, None]:
        return self.input_data.size_factors

    def export_params(self, append_to=None, **kwargs):
        if append_to is not None:
            if isinstance(append_to, anndata.AnnData):
                # append_to.obsm["design"] = self.design
                append_to.varm["a_var"] = np.transpose(self.a)
                append_to.varm["b_var"] = np.transpose(self.b)
            elif isinstance(append_to, xr.Dataset):
                # append_to["design"] = (self.param_shapes()["design"], self.design)
                append_to["a_var"] = (self.param_shapes()["a_var"], self.a)
                append_to["b_var"] = (self.param_shapes()["b_var"], self.b)
            else:
                raise ValueError("Unsupported data type: %s" % str(type(append_to)))
        else:
            ds = xr.Dataset({
                # "design": (self.param_shapes()["design"], self.design),
                "a_var": (self.param_shapes()["a_var"], self.a),
                "b_var": (self.param_shapes()["b_var"], self.b),
            })
            return ds

    @property
    @abc.abstractmethod
    def a_var(self) -> xr.DataArray:
        pass

    @property
    @abc.abstractmethod
    def b_var(self) -> xr.DataArray:
        pass

    @property
    def a(self) -> xr.DataArray:
        # TODO: take this out once xr.dataset slicing yields dataarray with loc_names coordinate:
        #return self.constraints_loc.dot(self.a_var, dims="loc_params")
        return np.matmul(self.constraints_loc.values, self.a_var.values)

    @property
    def b(self) -> xr.DataArray:
        # TODO: take this out once xr.dataset slicing yields dataarray with loc_names coordinate:
        #return self.constraints_scale.dot(self.b_var, dims="scale_params")
        return np.matmul(self.constraints_scale.values, self.b_var.values)

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


def _model_from_params(data: Union[xr.Dataset, anndata.AnnData, xr.DataArray], params=None, a=None, b=None):
    input_data = InputData.new(data)

    if params is None:
        if isinstance(data, Model):
            params = xr.Dataset({
                "a_var": data.a,
                "b_var": data.b,
            })
        elif anndata is not None and isinstance(data, anndata.AnnData):
            params = xr.Dataset({
                "a_var": (MODEL_PARAMS["a_var"], np.transpose(data.varm["a_var"])),
                "b_var": (MODEL_PARAMS["b_var"], np.transpose(data.varm["b_var"])),
            })
        elif isinstance(data, xr.Dataset):
            params = data
        else:
            params = xr.Dataset({
                "a_var": (MODEL_PARAMS["a_var"], a) if not isinstance(a, xr.DataArray) else a,
                "b_var": (MODEL_PARAMS["b_var"], b) if not isinstance(b, xr.DataArray) else b,
            })

    return input_data, params


class _Model_XArray_GLM(_Model_XArray_Base):
    _input_data: InputData
    params: xr.Dataset

    def __init__(self, input_data: InputData, params: xr.Dataset):
        super(_Model_XArray_Base, self).__init__(input_data=input_data, params=params)

    @property
    def a_var(self):
        return self.params["a_var"]

    @property
    def b_var(self):
        return self.params["b_var"]

    def __str__(self):
        return "[%s.%s object at %s]: data=%s" % (
            type(self).__module__,
            type(self).__name__,
            hex(id(self)),
            self.params
        )

    def __repr__(self):
        return self.__str__()
