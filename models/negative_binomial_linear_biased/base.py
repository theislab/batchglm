import abc
from typing import Union, Iterable

try:
    import anndata
except ImportError:
    anndata = None
import xarray as xr
import numpy as np

from models.negative_binomial.base import Model as NegativeBinomialModel
from models.negative_binomial.base import MODEL_PARAMS as NB_MODEL_PARAMS
from models.base import BasicEstimator

MODEL_PARAMS = NB_MODEL_PARAMS.copy()
MODEL_PARAMS.update({
    "design": ("samples", "design_params"),
    "a": ("design_params", "genes"),
    "b": ("design_params", "genes"),
})

ESTIMATOR_PARAMS = MODEL_PARAMS.copy()
ESTIMATOR_PARAMS.update({
    "loss": (),
    "gradient": ("genes",)
})


class Model(NegativeBinomialModel, metaclass=abc.ABCMeta):

    @classmethod
    def params(cls) -> dict:
        return MODEL_PARAMS

    @property
    @abc.abstractmethod
    def design(self) -> Union[np.ndarray, Iterable, int, float]:
        pass

    @property
    @abc.abstractmethod
    def a(self) -> Union[np.ndarray, Iterable, int, float]:
        pass

    @property
    @abc.abstractmethod
    def b(self) -> Union[np.ndarray, Iterable, int, float]:
        pass

    @property
    def mu(self):
        return np.exp(np.matmul(self.design, self.a))

    @property
    def r(self):
        return np.exp(np.matmul(self.design, self.b))

    def export_params(self, append_to=None, **kwargs):
        if append_to is not None:
            if isinstance(append_to, anndata.AnnData):
                append_to.obsm["design"] = self.design
                append_to.varm["a"] = np.transpose(self.a)
                append_to.varm["b"] = np.transpose(self.b)
            elif isinstance(append_to, xr.Dataset):
                append_to["design"] = (self.params()["design"], self.design)
                append_to["a"] = (self.params()["a"], self.a)
                append_to["b"] = (self.params()["b"], self.b)
            else:
                raise ValueError("Unsupported data type: %s" % str(type(append_to)))
        else:
            ds = xr.Dataset({
                "design": (self.params()["design"], self.design),
                "a": (self.params()["a"], self.a),
                "b": (self.params()["b"], self.b),
            })
            return ds


def model_from_params(data: Union[xr.Dataset, anndata.AnnData] = None, design=None, a=None, b=None) -> Model:
    if anndata is not None and isinstance(data, anndata.AnnData):
        return AnndataModel(data)
    elif isinstance(data, xr.Dataset):
        return XArrayModel(data)
    else:
        ds = xr.Dataset({
            "design": (MODEL_PARAMS["design"], design),
            "a": (MODEL_PARAMS["a"], a),
            "b": (MODEL_PARAMS["b"], b),
        })
        if data is not None:
            ds["sample_data"] = (MODEL_PARAMS["sample_data"], data)

        return XArrayModel(ds)


class XArrayModel(Model):
    data: xr.Dataset

    def __init__(self, data: xr.Dataset):
        self.data = data

    @property
    def sample_data(self):
        return self.data["sample_data"]

    @property
    def design(self) -> Union[np.ndarray, Iterable, int, float]:
        return self.data["design"]

    @property
    def a(self):
        return self.data['a']

    @property
    def b(self):
        return self.data['b']


class AnndataModel(Model):
    data: anndata.AnnData

    def __init__(self, data: anndata.AnnData):
        self.data = data

    @property
    def sample_data(self):
        return self.data.X

    @property
    def design(self):
        return self.data.obsm["design"]

    @property
    def a(self):
        return np.transpose(self.data.varm["a"])

    @property
    def b(self):
        return np.transpose(self.data.varm["b"])


class AbstractEstimator(Model, BasicEstimator, metaclass=abc.ABCMeta):

    @classmethod
    def params(cls) -> dict:
        return ESTIMATOR_PARAMS
