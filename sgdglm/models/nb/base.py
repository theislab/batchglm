import abc
from typing import Union

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

    @classmethod
    def from_data(cls, *args, from_store=False, **kwargs):
        return cls(*args, **kwargs, from_store=False)

    @classmethod
    def from_store(cls, *args, from_store=True, **kwargs):
        return cls(*args, **kwargs, from_store=True)

    def __init__(self, data, observation_names=None, feature_names=None, from_store=False):
        if from_store:
            self.data = data
            return

        # fetch_X = None
        # all_observations_zero = None
        if anndata is not None and isinstance(data, anndata.AnnData):
            X = data.X
            # all_observations_zero = ~np.any(data.X, axis=0)
            #
            # num_features = data.n_vars
            # num_observations = data.n_obs
            #
            # def fetch_X(idx):
            #     idx = np.asarray(idx).reshape(-1)
            #     retval = data.chunk_X(idx)[:, ~all_observations_zero]
            #
            #     if idx.size == 1:
            #         retval = np.squeeze(retval, axis=0)
            #
            #     return retval.astype(np.float32)
            #
            # delayed_fetch = dask.delayed(fetch_X, pure=True)
            # X = [
            #     dask.array.from_delayed(
            #         delayed_fetch(idx),
            #         shape=(num_features,),
            #         dtype=np.float32
            #     ) for idx in range(num_observations)
            # ]
            X = dask.array.from_array(X, X.shape)

            X = xr.DataArray(dask.array.stack(X), dims=INPUT_DATA_PARAMS["X"], coords={
                "observations": data.obs_names,
                "features": data.var_names,
            })
        elif isinstance(data, xr.Dataset):
            X: xr.DataArray = data["X"]
        elif isinstance(data, xr.DataArray):
            X = data
        else:
            X = xr.DataArray(data, dims=INPUT_DATA_PARAMS["X"])

        X = X.astype("float32")
        # X = X.chunk({"observations": 1})

        # if all_observations_zero is None:
        #     all_observations_zero = ~
        #
        # if fetch_X is None:
        #     def fetch_X(idx):
        #         return X[idx, ~all_observations_zero].values

        self.data = xr.Dataset({
            "X": X,
        }, coords={
            "feature_allzero": ~X.any(dim="observations")
        })
        if observation_names is not None:
            self.data = self.data.assign_coords(observations=observation_names)
        elif "observations" not in self.data.coords:
            self.data = self.data.assign_coords(observations=self.data.coords["observations"])

        if feature_names is not None:
            self.data = self.data.assign_coords(features=feature_names)
        elif "features" not in self.data.coords:
            self.data = self.data.assign_coords(features=self.data.coords["features"])

        # self.fetch_X = fetch_X

    @property
    def X(self):
        return self.data.X

    @X.setter
    def X(self, data):
        self.data["X"] = data

    @property
    def num_observations(self):
        return self.data.dims["observations"]

    @property
    def num_features(self):
        return self.data.dims["features"]

    @property
    def feature_isnonzero(self):
        return ~self.feature_isallzero

    @property
    def feature_isallzero(self):
        return self.data.coords["feature_allzero"]

    def fetch_X(self, idx):
        return self.X[idx].values

    def set_chunk_size(self, cs: int):
        self.X = self.X.chunk({"observations": cs})

    def __copy__(self):
        return self.from_store(self.data)

    def __getitem__(self, item):
        if isinstance(item, slice):
            data = self.data.isel(observations=item)
        elif isinstance(item, tuple):
            data = self.data.isel(observations=item[0], features=item[1])
        else:
            data = self.data.isel(observations=item)

        return self.from_store(data)

    def __str__(self):
        return "[%s.%s object at %s]: data=%s" % (type(self).__module__, type(self).__name__, hex(id(self)), self.data)

    def __repr__(self):
        return self.__str__()


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
    def feature_isnonzero(self):
        return self.input_data.feature_isnonzero

    @property
    def feature_isallzero(self):
        return self.input_data.feature_isallzero

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

    def __str__(self):
        return "[%s.%s object at %s]: params=%s" % \
               (type(self).__module__, type(self).__name__, hex(id(self)), self.params)

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
