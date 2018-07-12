import abc
from typing import Union

try:
    import anndata
except ImportError:
    anndata = None
import xarray as xr
import numpy as np

from models.nb.base import InputData as NegativeBinomialInputData
from models.nb.base import Model as NegativeBinomialModel
from models.nb.base import MODEL_PARAMS as NB_MODEL_PARAMS
from models.nb.base import INPUT_DATA_PARAMS as NB_INPUT_DATA_PARAMS
from models.base import BasicEstimator

INPUT_DATA_PARAMS = NB_INPUT_DATA_PARAMS.copy()
INPUT_DATA_PARAMS.update({
    "design": ("observations", "design_params"),
})

MODEL_PARAMS = NB_MODEL_PARAMS.copy()
MODEL_PARAMS.update({
    "a": ("design_params", "features"),
    "b": ("design_params", "features"),
})

ESTIMATOR_PARAMS = MODEL_PARAMS.copy()
ESTIMATOR_PARAMS.update({
    "loss": (),
    "gradient": ("features",),
    "hessian_diagonal": ("features", "variables",),
})


class InputData(NegativeBinomialInputData):

    def __init__(self, data, design=None, scaling_factors=None, observation_names=None, feature_names=None,
                 design_key="design", from_store=False):
        super().__init__(data=data, observation_names=observation_names, feature_names=feature_names,
                         from_store=from_store)
        if from_store:
            return

        if design is not None:
            if isinstance(design, xr.DataArray):
                self.design = design
            else:
                self.design = xr.DataArray(design, dims=INPUT_DATA_PARAMS["design"])
        elif anndata is not None and isinstance(data, anndata.AnnData):
            design = data.obsm[design_key]
            design = xr.DataArray(design, dims=INPUT_DATA_PARAMS["design"])
        elif isinstance(data, xr.Dataset):
            design: xr.DataArray = data[design_key]
        else:
            raise ValueError("Missing design matrix!")

        design = design.astype("float32")
        # design = design.chunk({"observations": 1})

        self.data["design"] = design

        if scaling_factors is not None:
            self.scaling_factors = scaling_factors

    @property
    def design(self):
        return self.data["design"]

    @design.setter
    def design(self, data):
        self.data["design"] = data

    @property
    def scaling_factors(self):
        return self.data.coords.get("scaling_factors")

    @scaling_factors.setter
    def scaling_factors(self, data):
        self.data.assign_coords(scaling_factors=data)

    @property
    def num_design_params(self):
        return self.data.dims["design_params"]

    def fetch_design(self, idx):
        return self.design[idx]

    def fetch_scaling_factors(self, idx):
        return self.scaling_factors[idx]

    def set_chunk_size(self, cs: int):
        super().set_chunk_size(cs)
        self.design = self.design.chunk({"observations": cs})


class Model(NegativeBinomialModel, metaclass=abc.ABCMeta):

    @classmethod
    def param_shapes(cls) -> dict:
        return MODEL_PARAMS

    @property
    @abc.abstractmethod
    def input_data(self) -> InputData:
        pass

    @property
    def design(self) -> xr.DataArray:
        return self.input_data.design

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
        return np.exp(self.design.dot(self.a))

    @property
    def r(self) -> xr.DataArray:
        return np.exp(self.design.dot(self.b))

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
    input_data = InputData(data)

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
        params = estim.to_xarray(["a", "b", "loss", "gradient", "hessian_diagonal"])

        XArrayModel.__init__(self, input_data, params)

    @property
    def input_data(self):
        return self._input_data

    @property
    def loss(self):
        return self.params["loss"]

    @property
    def gradient(self):
        return self.params["gradient"]

    @property
    def hessian_diagonal(self):
        return self.params["hessian_diagonal"]
