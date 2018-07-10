import abc
from typing import Union, Iterable, Type

import dask
import dask.array as da

try:
    import anndata
except ImportError:
    anndata = None
import xarray as xr
import numpy as np

from models.base import BasicInputData, BasicModel, BasicEstimator
import utils.random as rand_utils

INPUT_DATA_PARAMS = {
    "X": ("observations", "features"),
}

MODEL_PARAMS = INPUT_DATA_PARAMS.copy()
MODEL_PARAMS.update({
    "mu": ("observations", "features"),
    "r": ("observations", "features"),
    "sigma2": ("observations", "features"),
    "probs": ("observations", "features"),
    "log_probs": ("observations", "features"),
    "log_likelihood": (),
})

ESTIMATOR_PARAMS = MODEL_PARAMS.copy()
ESTIMATOR_PARAMS.update({
    "loss": (),
    "gradient": ("features",),
    "hessian_diagonal": ("features", "variables",),
})


class InputData(BasicInputData):

    def __init__(self, data):
        fetch_X = None
        if anndata is not None and isinstance(data, anndata.AnnData):
            X = data.X

            num_features = data.n_vars
            num_observations = data.n_obs

            def fetch_X(idx):
                idx = np.asarray(idx).reshape(-1)
                retval = data.chunk_X(idx)
                if idx.size == 1:
                    retval = np.squeeze(retval, axis=0)
                return retval.astype(np.float32)

            delayed_fetch = dask.delayed(fetch_X, pure=True)
            X = [
                dask.array.from_delayed(
                    delayed_fetch(idx),
                    shape=(num_features,),
                    dtype=np.float32
                ) for idx in range(num_observations)
            ]

            X = xr.DataArray(dask.array.stack(X), dims=INPUT_DATA_PARAMS["X"])
        elif isinstance(data, xr.Dataset):
            X: xr.DataArray = data["X"]
        elif isinstance(data, xr.DataArray):
            X = data
        else:
            X = xr.DataArray(data, dims=INPUT_DATA_PARAMS["X"])

        (num_observations, num_features) = X.shape

        X = X.astype("float32")
        # X = X.chunk({"observations": 1})
        if fetch_X is None:
            def fetch_X(idx):
                return X[idx].values

        self.X = X
        self.num_observations = num_observations
        self.num_features = num_features

        self.fetch_X = fetch_X

    def set_chunk_size(self, cs: int):
        self.X = self.X.chunk({"observations": cs})

    def __copy__(self):
        X = self.X.copy()
        return InputData(data=X)


class Model(BasicModel, metaclass=abc.ABCMeta):

    @classmethod
    def param_shapes(cls) -> dict:
        return MODEL_PARAMS

    @property
    @abc.abstractmethod
    def input_data(self) -> InputData:
        pass

    @property
    def X(self) -> xr.DataArray:
        return self.input_data.X

    @property
    @abc.abstractmethod
    def mu(self) -> xr.DataArray:
        pass

    @property
    @abc.abstractmethod
    def r(self) -> xr.DataArray:
        pass

    @property
    def sigma2(self) -> xr.DataArray:
        return self.mu + ((self.mu * self.mu) / self.r)

    def probs(self) -> xr.DataArray:
        X = self.X
        return rand_utils.NegativeBinomial(mean=self.mu, r=self.r).prob(X)

    def log_probs(self) -> xr.DataArray:
        X = self.X
        return rand_utils.NegativeBinomial(mean=self.mu, r=self.r).log_prob(X)

    def log_likelihood(self) -> xr.DataArray:
        retval: xr.DataArray = np.sum(self.log_probs())
        return retval

    def export_params(self, append_to=None, **kwargs):
        """
        Export model parameters as xr.Dataset or append it to some xr.Dataset or anndata.Anndata

        :param append_to: xr.Dataset or anndata.Anndata.

            Parameters will be appended to this, if specified.
        :return: xr.Dataset or None, if `append_to` was specified
        """
        if append_to is not None:
            if isinstance(append_to, anndata.AnnData):
                append_to.obsm["mu"] = self.mu
                append_to.obsm["r"] = self.r
            elif isinstance(append_to, xr.Dataset):
                append_to["mu"] = (self.param_shapes()["mu"], self.mu)
                append_to["r"] = (self.param_shapes()["r"], self.r)
            else:
                raise ValueError("Unsupported data type: %s" % str(type(append_to)))
        else:
            ds = xr.Dataset({
                "mu": (self.param_shapes()["mu"], self.mu),
                "r": (self.param_shapes()["r"], self.r),
            })
            return ds


def _model_from_params(data: Union[xr.Dataset, anndata.AnnData, xr.DataArray], params=None, mu=None, r=None):
    if isinstance(data, Model):
        input_data = data.input_data
    else:
        input_data = InputData(data)

    if params is None:
        if isinstance(data, Model):
            params = xr.Dataset({
                "mu": data.mu,
                "r": data.r,
            })
        elif anndata is not None and isinstance(data, anndata.AnnData):
            params = xr.Dataset({
                "mu": (MODEL_PARAMS["mu"], data.obsm["mu"]),
                "r": (MODEL_PARAMS["r"], data.obsm["r"]),
            })
        elif isinstance(data, xr.Dataset):
            params = data
        else:
            params = xr.Dataset({
                "mu": (MODEL_PARAMS["mu"], mu) if not isinstance(mu, xr.DataArray) else mu,
                "r": (MODEL_PARAMS["r"], r) if not isinstance(r, xr.DataArray) else r,
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
    def mu(self):
        return self.params["mu"]

    @property
    def r(self):
        return self.params["r"]


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
#     def mu(self):
#         return self.data.obsm["mu"]
#
#     @property
#     def r(self):
#         return self.data.obsm["r"]


class AbstractEstimator(Model, BasicEstimator, metaclass=abc.ABCMeta):
    input_data: InputData

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
        params = estim.to_xarray(["mu", "r", "loss", "gradient", "hessian_diagonal"])

        XArrayModel.__init__(self, input_data, params)

    @property
    def input_data(self):
        return self._input_data

    @property
    def loss(self, **kwargs):
        return self.params["loss"]

    @property
    def gradient(self, **kwargs):
        return self.params["gradient"]

    @property
    def hessian_diagonal(self):
        return self.params["hessian_diagonal"]
