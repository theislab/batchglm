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

from ..base import BasicInputData, BasicModel, BasicEstimator
from ..external import rand_utils, data_utils

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
    """
    Input data holder for negative binomial distributed data.
    """
    data: xr.Dataset

    @classmethod
    def param_shapes(cls) -> dict:
        return INPUT_DATA_PARAMS

    @classmethod
    def new(cls, data, observation_names=None, feature_names=None, cast_dtype=None):
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
        :param observation_names: (optional) names of the observations.
        :param feature_names: (optional) names of the features.
        :param cast_dtype: data type of all data; should be either float32 or float64
        :return: InputData object
        """
        X = data_utils.xarray_from_data(data)

        if cast_dtype is not None:
            X = X.astype(cast_dtype)
            # X = X.chunk({"observations": 1})

        retval = cls(xr.Dataset({
            "X": X,
        }, coords={
            "feature_allzero": ~X.any(dim="observations")
        }))
        if observation_names is not None:
            retval.observations = observation_names
        elif "observations" not in retval.data.coords:
            retval.observations = retval.data.coords["observations"]

        if feature_names is not None:
            retval.features = feature_names
        elif "features" not in retval.data.coords:
            retval.features = retval.data.coords["features"]

        return retval

    @property
    def X(self) -> xr.DataArray:
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
    def observations(self):
        return self.data.coords["observations"]

    @observations.setter
    def observations(self, data):
        self.data.coords["observations"] = data

    @property
    def features(self):
        return self.data.coords["features"]

    @features.setter
    def features(self, data):
        self.data.coords["features"] = data

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
        return type(self)(self.data)

    def __getitem__(self, item):
        if isinstance(item, slice):
            data = self.data.isel(observations=item)
        elif isinstance(item, tuple):
            data = self.data.isel(observations=item[0], features=item[1])
        else:
            data = self.data.isel(observations=item)

        return type(self)(data)

    def __str__(self):
        return "[%s.%s object at %s]: data=%s" % (
            type(self).__module__,
            type(self).__name__,
            hex(id(self)),
            self.data
        )

    def __repr__(self):
        return self.__str__()


class Model(BasicModel, metaclass=abc.ABCMeta):
    """
    Negative binomial model
    """

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
    def num_observations(self):
        return self.input_data.num_observations

    @property
    def num_features(self):
        return self.input_data.num_features

    @property
    def observations(self):
        return self.input_data.observations

    @property
    def features(self):
        return self.input_data.features

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
        input_data = InputData.new(data)

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
