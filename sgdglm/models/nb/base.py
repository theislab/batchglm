import abc
from typing import Union, Iterable

try:
    import anndata
except ImportError:
    anndata = None
import xarray as xr
import numpy as np

from models.base import BasicModel, BasicEstimator
import utils.random as rand_utils

MODEL_PARAMS = {
    "X": ("observations", "features"),
    "mu": ("observations", "features"),
    "r": ("observations", "features"),
    "sigma2": ("observations", "features"),
    "probs": ("observations", "features"),
    "log_probs": ("observations", "features"),
    "log_likelihood": (),
}

ESTIMATOR_PARAMS = MODEL_PARAMS.copy()
ESTIMATOR_PARAMS.update({
    "loss": (),
})


class Model(BasicModel, metaclass=abc.ABCMeta):

    @classmethod
    def params(cls) -> dict:
        return MODEL_PARAMS

    @property
    def X(self) -> Union[np.ndarray, Iterable, int, float]:
        raise ValueError("no sample data specified")

    @property
    @abc.abstractmethod
    def mu(self) -> Union[np.ndarray, Iterable, int, float]:
        pass

    @property
    @abc.abstractmethod
    def r(self) -> Union[np.ndarray, Iterable, int, float]:
        pass

    @property
    def sigma2(self) -> Union[np.ndarray, Iterable, int, float]:
        return self.mu + ((self.mu * self.mu) / self.r)

    def probs(self, X=None) -> Union[np.ndarray, Iterable, int, float]:
        if X is None:
            X = self.X
        return rand_utils.NegativeBinomial(mean=self.mu, r=self.r).prob(X)

    def log_probs(self, X=None) -> Union[np.ndarray, Iterable, int, float]:
        if X is None:
            X = self.X
        return rand_utils.NegativeBinomial(mean=self.mu, r=self.r).log_prob(X)

    def log_likelihood(self, X=None) -> Union[np.ndarray, Iterable, int, float]:
        if X is None:
            X = self.X
        return np.sum(self.log_probs(X))

    def export_params(self, append_to=None, **kwargs):
        if append_to is not None:
            if isinstance(append_to, anndata.AnnData):
                append_to.obsm["mu"] = self.mu
                append_to.obsm["r"] = self.r
            elif isinstance(append_to, xr.Dataset):
                append_to["mu"] = (self.params()["mu"], self.mu)
                append_to["r"] = (self.params()["r"], self.r)
            else:
                raise ValueError("Unsupported data type: %s" % str(type(append_to)))
        else:
            ds = xr.Dataset({
                "mu": (self.params()["mu"], self.mu),
                "r": (self.params()["r"], self.r),
            })
            return ds


def model_from_params(data: Union[xr.Dataset, anndata.AnnData] = None, mu=None, r=None) -> Model:
    if anndata is not None and isinstance(data, anndata.AnnData):
        return AnndataModel(data)
    elif isinstance(data, xr.Dataset):
        return XArrayModel(data)
    else:
        ds = xr.Dataset({
            "mu": (MODEL_PARAMS["mu"], mu),
            "r": (MODEL_PARAMS["r"], r),
        })
        if data is not None:
            ds["X"] = (MODEL_PARAMS["X"], data)

        return XArrayModel(ds)


class XArrayModel(Model):
    data: xr.Dataset

    def __init__(self, data: xr.Dataset):
        self.data = data

    @property
    def X(self):
        return self.data["X"].values

    @property
    def mu(self):
        return self.data["mu"].values

    @property
    def r(self):
        return self.data["r"].values


class AnndataModel(Model):
    data: anndata.AnnData

    def __init__(self, data: anndata.AnnData):
        self.data = data

    @property
    def X(self):
        return self.data.X

    @property
    def mu(self):
        return self.data.obsm["mu"]

    @property
    def r(self):
        return self.data.obsm["r"]


class AbstractEstimator(Model, BasicEstimator, metaclass=abc.ABCMeta):
    input_data: Union[xr.Dataset, anndata.AnnData]

    @classmethod
    def params(cls) -> dict:
        return ESTIMATOR_PARAMS
