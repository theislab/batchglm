import abc
try:
    import anndata
except ImportError:
    anndata = None
import xarray as xr
import numpy as np

from .external import InputData
from .external import _Model_GLM, _Model_XArray_GLM, MODEL_PARAMS, _model_from_params

# Define distribution parameters:
MODEL_PARAMS = MODEL_PARAMS.copy()
MODEL_PARAMS.update({
    "mean": ("observations", "features"),
    "sd": ("observations", "features"),
})

class Model(_Model_GLM, metaclass=abc.ABCMeta):
    """
    Generalized Linear Model (GLM) with normal noise.
    """

    @classmethod
    def param_shapes(cls) -> dict:
        return MODEL_PARAMS

    def link_loc(self, data):
        return data

    def inverse_link_loc(self, data):
        return data

    def link_scale(self, data):
        return np.log(data)

    def inverse_link_scale(self, data):
        return np.exp(data)

    @property
    def eta_loc(self) -> xr.DataArray:
        # TODO: take this switch out once xr.dataset slicing yields dataarray with loc_names coordinate:
        if isinstance(self.par_link_loc, xr.DataArray):
            eta = self.design_loc.dot(self.par_link_loc, dims="design_loc_params")
        else:
            eta = np.matmul(self.design_loc.values, self.par_link_loc)

        if self.size_factors is not None:
            eta *= np.expand_dims(self.size_factors, axis=1)
        return eta

    @property
    def mean(self) -> xr.DataArray:
        return self.location

    @property
    def sd(self) -> xr.DataArray:
        return self.scale


def model_from_params(*args, **kwargs) -> Model:
    (input_data, params) = _model_from_params(*args, **kwargs)
    return Model_XArray(input_data, params)


class Model_XArray(_Model_XArray_GLM, Model):
    _input_data: InputData
    params: xr.Dataset

    def __init__(self, input_data: InputData, params: xr.Dataset):
        super(_Model_XArray_GLM, self).__init__(input_data=input_data, params=params)
        super(Model, self).__init__()

    def __str__(self):
        return "[%s.%s object at %s]: data=%s" % (
            type(self).__module__,
            type(self).__name__,
            hex(id(self)),
            self.params
        )

    def __repr__(self):
        return self.__str__()
