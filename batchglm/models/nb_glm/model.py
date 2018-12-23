import abc
from typing import Union

try:
    import anndata
except ImportError:
    anndata = None
import xarray as xr
import numpy as np

from .input import InputData, INPUT_DATA_PARAMS
from .external import BasicGLM, BasicModel

# Define distribution parameters:
MODEL_PARAMS = INPUT_DATA_PARAMS.copy()
MODEL_PARAMS.update({
    "mu": ("observations", "features"),
    "r": ("observations", "features"),
    "sigma2": ("observations", "features"),
    "probs": ("observations", "features"),
    "log_probs": ("observations", "features"),
    "log_likelihood": (),
    "a": ("design_loc_params", "features"),
    "b": ("design_scale_params", "features"),
    "par_link_loc": ("design_loc_params", "features"),
    "par_link_scale": ("design_scale_params", "features"),
})

class Model(BasicGLM, BasicModel, metaclass=abc.ABCMeta):
    """
    Generalized Linear Model (GLM) with negative binomial noise.
    """

    @classmethod
    def param_shapes(cls) -> dict:
        return MODEL_PARAMS

    @property
    @abc.abstractmethod
    def input_data(self) -> InputData:
        pass

    @property
    def design_loc(self) -> xr.DataArray:
        return self.input_data.design_loc

    @property
    def design_scale(self) -> xr.DataArray:
        return self.input_data.design_scale

    @property
    def par_link_loc(self):
        return self.a

    @property
    def par_link_scale(self):
        return self.b

    def link_loc(self, data):
        return np.log(data)

    def inverse_link_loc(self, data):
        return np.exp(data)

    def link_scale(self, data):
        return np.log(data)

    def inverse_link_scale(self, data):
        return np.exp(data)

    @property
    def location(self):
        return self.mu

    @property
    def scale(self):
        return self.r

    @property
    def size_factors(self) -> Union[xr.DataArray, None]:
        return self.input_data.size_factors

    @property
    @abc.abstractmethod
    def a(self) -> xr.DataArray:
        pass

    @property
    @abc.abstractmethod
    def b(self) -> xr.DataArray:
        pass

    @property
    def mu(self) -> xr.DataArray:
        # exp(design * a + sf)
        # Depend on xarray and use xr.DataArray.dot() for matrix multiplication
        # Also, make sure that the right order of dimensions get returned => use xr.DataArray.transpose()
        log_retval = self.design_loc.dot(self.par_link_loc, dims="design_loc_params")
        log_retval = log_retval.transpose(*self.param_shapes()["mu"])

        if self.size_factors is not None:
            log_retval += self.link_loc(self.size_factors)

        return np.exp(log_retval)

    @property
    def r(self) -> xr.DataArray:
        # exp(design * b)
        # Depend on xarray and use xr.DataArray.dot() for matrix multiplication
        # Also, make sure that the right order of dimensions get returned => use xr.DataArray.transpose()
        log_retval = self.design_scale.dot(self.par_link_scale, dims="design_scale_params")
        log_retval = log_retval.transpose(*self.param_shapes()["r"])

        return np.exp(log_retval)

    def export_params(self, append_to=None, **kwargs):
        if append_to is not None:
            if isinstance(append_to, anndata.AnnData):
                # append_to.obsm["design"] = self.design
                append_to.varm["a"] = np.transpose(self.a)
                append_to.varm["b"] = np.transpose(self.b)
            elif isinstance(append_to, xr.Dataset):
                # append_to["design"] = (self.param_shapes()["design"], self.design)
                append_to["a"] = (self.param_shapes()["a"], self.a)
                append_to["b"] = (self.param_shapes()["b"], self.b)
            else:
                raise ValueError("Unsupported data type: %s" % str(type(append_to)))
        else:
            ds = xr.Dataset({
                # "design": (self.param_shapes()["design"], self.design),
                "a": (self.param_shapes()["a"], self.a),
                "b": (self.param_shapes()["b"], self.b),
            })
            return ds


def _model_from_params(data: Union[xr.Dataset, anndata.AnnData, xr.DataArray], params=None, a=None, b=None):
    input_data = InputData.new(data)

    if params is None:
        if isinstance(data, Model):
            params = xr.Dataset({
                "a": data.a,
                "b": data.b,
            })
        elif anndata is not None and isinstance(data, anndata.AnnData):
            params = xr.Dataset({
                "a": (MODEL_PARAMS["a"], np.transpose(data.varm["a"])),
                "b": (MODEL_PARAMS["b"], np.transpose(data.varm["b"])),
            })
        elif isinstance(data, xr.Dataset):
            params = data
        else:
            params = xr.Dataset({
                "a": (MODEL_PARAMS["a"], a) if not isinstance(a, xr.DataArray) else a,
                "b": (MODEL_PARAMS["b"], b) if not isinstance(b, xr.DataArray) else b,
            })

    return input_data, params


def model_from_params(*args, **kwargs) -> Model:
    (input_data, params) = _model_from_params(*args, **kwargs)
    return XArrayModel(input_data, params)


class XArrayModel(Model):
    _input_data: InputData
    params: xr.Dataset

    def __init__(self, input_data: InputData, params: xr.Dataset):
        self._input_data = input_data
        self.params = params

    @property
    def input_data(self) -> InputData:
        return self._input_data

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
