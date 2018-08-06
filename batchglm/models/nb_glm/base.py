import abc
from typing import Union

try:
    import anndata
except ImportError:
    anndata = None
import xarray as xr
import numpy as np
import pandas as pd

from ..nb.base import InputData as NegativeBinomialInputData
from ..nb.base import Model as NegativeBinomialModel
from ..nb.base import MODEL_PARAMS as NB_MODEL_PARAMS
from ..nb.base import INPUT_DATA_PARAMS as NB_INPUT_DATA_PARAMS
from ..base import BasicEstimator
from ..glm import Model as GeneralizedLinearModel

import patsy

INPUT_DATA_PARAMS = NB_INPUT_DATA_PARAMS.copy()
INPUT_DATA_PARAMS.update({
    "design_loc": ("observations", "design_loc_params"),
    "design_scale": ("observations", "design_scale_params"),
})

MODEL_PARAMS = NB_MODEL_PARAMS.copy()
MODEL_PARAMS.update(INPUT_DATA_PARAMS)
MODEL_PARAMS.update({
    "a": ("design_loc_params", "features"),
    "b": ("design_scale_params", "features"),
})

ESTIMATOR_PARAMS = MODEL_PARAMS.copy()
ESTIMATOR_PARAMS.update({
    "loss": (),
    "gradient": ("features",),
    "hessians": ("features", "delta_var0", "delta_var1"),
    "fisher_inv": ("features", "delta_var0", "delta_var1"),
})


def _parse_design(data, design, names, design_key, dim="design_params"):
    if design is not None:
        if isinstance(design, patsy.design_info.DesignMatrix):
            dmat = xr.DataArray(design, dims=("observations", dim))
            dmat.coords[dim] = design.design_info.column_names
        elif isinstance(design, xr.DataArray):
            dmat = design
        elif isinstance(design, pd.DataFrame):
            dmat = xr.DataArray(np.asarray(design), dims=("observations", dim))
            dmat.coords[dim] = design.columns
        else:
            dmat = xr.DataArray(design, dims=("observations", dim))
    elif anndata is not None and isinstance(data, anndata.AnnData):
        dmat = data.obsm[design_key]
        dmat = xr.DataArray(dmat, dims=("observations", dim))
    elif isinstance(data, xr.Dataset):
        dmat: xr.DataArray = data[design_key]
    else:
        raise ValueError("Missing design_loc matrix!")
    
    if names is not None:
        dmat.coords[dim] = names
    elif dim not in dmat.coords:
        dmat.coords[dim] = dmat.coords[dim]
        # raise ValueError("Could not find names for %s; Please specify them manually." % dim)
    
    return dmat


class InputData(NegativeBinomialInputData):
    
    @classmethod
    def new(
            cls,
            data: Union[np.ndarray, anndata.AnnData, xr.DataArray, xr.Dataset],
            design_loc: Union[np.ndarray, pd.DataFrame, patsy.design_info.DesignMatrix, xr.DataArray] = None,
            design_loc_names: Union[list, np.ndarray, xr.DataArray] = None,
            design_scale: Union[np.ndarray, pd.DataFrame, patsy.design_info.DesignMatrix, xr.DataArray] = None,
            design_scale_names: Union[list, np.ndarray, xr.DataArray] = None,
            size_factors=None,
            observation_names=None,
            feature_names=None,
            design_loc_key="design_loc",
            design_scale_key="design_scale",
    ):
        """
        Create a new InputData object.

        :param data: Some data object.

        Can be either:
            - np.ndarray: NumPy array containing the raw data
            - anndata.AnnData: AnnData object containing the count data and optional the design models
                stored as data.obsm[design_loc] and data.obsm[design_scale]
            - xr.DataArray: DataArray of shape ("observations", "features") containing the raw data
            - xr.Dataset: Dataset containing the raw data as data["X"] and optional the design models
                stored as data[design_loc] and data[design_scale]
        :param design_loc: Some matrix containing the location design model.
            Optional, if already specified in `data`
        :param design_loc_names: (optional) names of the design_loc parameters.
            The names might already be included in `design_loc`.
            Will be used to find identical columns in two models.
        :param design_scale: Some matrix containing the scale design model.
            Optional, if already specified in `data`
        :param design_scale_names: (optional) names of the design_scale parameters.
            The names might already be included in `design_loc`.
            Will be used to find identical columns in two models.
        :param size_factors: Some size factor to scale the raw data in link-space.
        :param observation_names: (optional) names of the observations.
        :param feature_names: (optional) names of the features.
        :param design_loc_key: Where to find `design_loc` if `data` is some anndata.AnnData or xarray.Dataset.
        :param design_scale_key: Where to find `design_scale` if `data` is some anndata.AnnData or xarray.Dataset.
        :return: InputData object
        """
        retval = super(InputData, cls).new(
            data=data,
            observation_names=observation_names,
            feature_names=feature_names
        )
        
        design_loc = _parse_design(data, design_loc, design_loc_names, design_loc_key, "design_loc_params")
        design_scale = _parse_design(data, design_scale, design_scale_names, design_scale_key, "design_scale_params")
        
        design_loc = design_loc.astype("float32")
        design_scale = design_scale.astype("float32")
        # design = design.chunk({"observations": 1})
        
        retval.design_loc = design_loc
        retval.design_scale = design_scale
        
        if size_factors is not None:
            retval.size_factors = size_factors
        
        return retval
    
    @property
    def design_loc(self) -> xr.DataArray:
        return self.data["design_loc"]
    
    @design_loc.setter
    def design_loc(self, data):
        self.data["design_loc"] = data
    
    @property
    def design_loc_names(self) -> xr.DataArray:
        return self.data.coords["design_loc_params"]
    
    @design_loc_names.setter
    def design_loc_names(self, data):
        self.data.coords["design_loc_params"] = data
    
    @property
    def design_scale(self) -> xr.DataArray:
        return self.data["design_scale"]
    
    @design_scale.setter
    def design_scale(self, data):
        self.data["design_scale"] = data
    
    @property
    def design_scale_names(self) -> xr.DataArray:
        return self.data.coords["design_scale_params"]
    
    @design_scale_names.setter
    def design_scale_names(self, data):
        self.data.coords["design_scale_params"] = data
    
    @property
    def size_factors(self):
        return self.data.coords.get("size_factors")
    
    @size_factors.setter
    def size_factors(self, data):
        self.data.coords["size_factors"] = data
    
    @property
    def num_design_loc_params(self):
        return self.data.dims["design_loc_params"]
    
    @property
    def num_design_scale_params(self):
        return self.data.dims["design_scale_params"]
    
    def fetch_design_loc(self, idx):
        return self.design_loc[idx]
    
    def fetch_design_scale(self, idx):
        return self.design_scale[idx]
    
    def fetch_size_factors(self, idx):
        return self.size_factors[idx]
    
    def set_chunk_size(self, cs: int):
        super().set_chunk_size(cs)
        self.design_loc = self.design_loc.chunk({"observations": cs})
        self.design_scale = self.design_scale.chunk({"observations": cs})


class Model(GeneralizedLinearModel, NegativeBinomialModel, metaclass=abc.ABCMeta):
    
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
    @abc.abstractmethod
    def a(self) -> xr.DataArray:
        pass
    
    @property
    @abc.abstractmethod
    def b(self) -> xr.DataArray:
        pass
    
    @property
    def mu(self) -> xr.DataArray:
        return np.exp(self.design_loc.dot(self.a))
    
    @property
    def r(self) -> xr.DataArray:
        return np.exp(self.design_scale.dot(self.b))
    
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
                "a": (MODEL_PARAMS["a"], data.obsm["a"]),
                "b": (MODEL_PARAMS["b"], data.obsm["b"]),
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


# class AnndataModel(Model):
#     data: anndata.AnnData
#
#     def __init__(self, data: anndata.AnnData):
#         self.data = data
#
#     @property
#     def X(self):
#         return self.data.X
#
#     @property
#     def design(self):
#         return self.data.obsm["design"]
#
#     @property
#     def a(self):
#         return np.transpose(self.data.varm["a"])
#
#     @property
#     def b(self):
#         return np.transpose(self.data.varm["b"])


class AbstractEstimator(Model, BasicEstimator, metaclass=abc.ABCMeta):
    
    @classmethod
    def param_shapes(cls) -> dict:
        return ESTIMATOR_PARAMS


class XArrayEstimatorStore(AbstractEstimator, XArrayModel):
    
    def initialize(self, **kwargs):
        raise NotImplementedError("This object only stores estimated values")
    
    def train(self, **kwargs):
        raise NotImplementedError("This object only stores estimated values")
    
    def finalize(self, **kwargs) -> AbstractEstimator:
        return self
    
    def validate_data(self, **kwargs):
        raise NotImplementedError("This object only stores estimated values")
    
    def __init__(self, estim: AbstractEstimator):
        input_data = estim.input_data
        params = estim.to_xarray(["a", "b", "loss", "gradient", "hessians", "fisher_inv"], coords=input_data.data)
        
        XArrayModel.__init__(self, input_data, params)
    
    @property
    def input_data(self):
        return self._input_data
    
    @property
    def loss(self):
        return self.params["loss"]
    
    @property
    def gradient(self):
        return self.params["loss"]
    
    @property
    def hessians(self):
        return self.params["hessians"]
    
    @property
    def fisher_inv(self):
        return self.params["fisher_inv"]
