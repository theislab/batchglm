import abc
from typing import Union

try:
    import anndata
except ImportError:
    anndata = None

import xarray as xr
import numpy as np
import pandas as pd
import patsy

import batchglm.pkg_constants as pkg_constants
import batchglm.utils.random as rand_utils
from batchglm.utils.numeric import softmax
from ..base import BasicEstimator
from ..glm import parse_design
from ..mixture import Model as MixtureModel
from ..nb_glm.base import InputData as NB_GLM_InputData
from ..nb_glm.base import Model as NB_GLM_Model
from ..nb_glm.base import MODEL_PARAMS as NB_GLM_MODEL_PARAMS
from ..nb_glm.base import INPUT_DATA_PARAMS as NB_GLM_INPUT_DATA_PARAMS

INPUT_DATA_PARAMS = NB_GLM_INPUT_DATA_PARAMS.copy()
INPUT_DATA_PARAMS.update({
    "design_loc": ("observations", "design_loc_params"),
    "design_scale": ("observations", "design_scale_params"),
    "design_mixture_loc": ("design_loc_params", "mixtures", "design_mixture_loc_params"),
    "design_mixture_scale": ("design_scale_params", "mixtures", "design_mixture_scale_params"),
    "mixture_weight_constraints": ("observations", "mixtures"),
})

MODEL_PARAMS = NB_GLM_MODEL_PARAMS.copy()
MODEL_PARAMS.update(INPUT_DATA_PARAMS)
MODEL_PARAMS.update({
    "a": ("design_loc_params", "design_mixture_loc_params", "features"),
    "b": ("design_scale_params", "design_mixture_scale_params", "features"),
    "par_link_loc": ("mixtures", "design_loc_params", "features"),
    "par_link_scale": ("mixtures", "design_scale_params", "features"),
    "mixture_assignment": ("observations",),
    "mixture_prob": ("observations", "mixtures"),
    "mixture_log_prob": ("observations", "mixtures"),
    "mu": ("mixtures", "observations", "features"),
    "r": ("mixtures", "observations", "features"),
    "sigma2": ("mixtures", "observations", "features"),
    "probs": ("mixtures", "observations", "features"),
    "log_probs": ("mixtures", "observations", "features"),
    "log_likelihood": (),
})

ESTIMATOR_PARAMS = MODEL_PARAMS.copy()
ESTIMATOR_PARAMS.update({
    "loss": (),
    "gradient": ("features",),
    # "hessians": ("features", "delta_var0", "delta_var1"),
    # "fisher_inv": ("features", "delta_var0", "delta_var1"),
})


def param_bounds(dtype: np.dtype, dmin=None, dmax=None):
    dtype = np.dtype(dtype)
    if dmin is None:
        dmin = np.finfo(dtype).min
    if dmax is None:
        dmax = np.finfo(dtype).max
    dtype = dtype.type

    sf = dtype(pkg_constants.ACCURACY_MARGIN_RELATIVE_TO_LIMIT)
    bounds_min = {
        "a": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
        "b": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
        "log_mu": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
        "log_r": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
        # "mu": np.nextafter(0, np.inf, dtype=dtype),
        # "r": np.nextafter(0, np.inf, dtype=dtype),
        # "probs": dtype(0),
        "probs": np.nextafter(0, np.inf, dtype=dtype),
        "log_probs": np.log(np.nextafter(0, np.inf, dtype=dtype)),
        "mixture_prob": np.nextafter(0, np.inf, dtype=dtype),
        "mixture_log_prob": np.log(np.nextafter(0, np.inf, dtype=dtype)),
        "mixture_logits": np.log(np.nextafter(0, np.inf, dtype=dtype)),
        # "mixture_weight_constraints": np.nextafter(0, np.inf, dtype=dtype),
        "mixture_weight_constraints": dtype(0),
        # "mixture_weight_log_constraints": np.log(np.nextafter(0, np.inf, dtype=dtype)),
        "mixture_weight_log_constraints": -np.sqrt(dmax),
    }
    bounds_min.update({
        "mu" : np.exp(bounds_min["log_mu"]),
        "r" : np.exp(bounds_min["log_r"]),
    })
    bounds_max = {
        "a": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
        "b": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
        "log_mu": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
        "log_r": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
        # "mu": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
        # "r": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
        "probs": dtype(1),
        "log_probs": dtype(0),
        "mixture_prob": dtype(1),
        "mixture_log_prob": dtype(0),
        "mixture_logits": dtype(0),
        "mixture_weight_constraints": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
        "mixture_weight_log_constraints": np.log(np.nextafter(np.inf, -np.inf, dtype=dtype)) / sf,
    }
    bounds_max.update({
        "mu" : np.exp(bounds_max["log_mu"]),
        "r" : np.exp(bounds_max["log_r"]),
    })

    return bounds_min, bounds_max


def np_clip_param(param, name):
    bounds_min, bounds_max = param_bounds(param.dtype)
    if isinstance(param, xr.DataArray):
        return param.clip(
            bounds_min[name],
            bounds_max[name],
            # out=param
        )
    else:
        return np.clip(
            param,
            bounds_min[name],
            bounds_max[name],
            # out=param
        )


class InputData(NB_GLM_InputData):
    """
    Input data for Generalized Linear Mixture Models (GLMMs) with negative binomial noise.
    """

    @classmethod
    def param_shapes(cls) -> dict:
        return INPUT_DATA_PARAMS

    @classmethod
    def new(
            cls,
            data,
            *args,
            design_mixture_loc: Union[np.ndarray, pd.DataFrame, patsy.design_info.DesignMatrix, xr.DataArray] = None,
            design_mixture_loc_key: str = "design_mixture_loc",
            design_mixture_scale: Union[np.ndarray, pd.DataFrame, patsy.design_info.DesignMatrix, xr.DataArray] = None,
            design_mixture_scale_key: str = "design_mixture_scale",
            mixture_weight_constraints=None,
            cast_dtype=None,
            **kwargs
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
        :param cast_dtype: If this option is set, all provided data will be casted to this data type.
        :return: InputData object
        """
        retval = super(InputData, cls).new(
            data,
            *args,
            cast_dtype=cast_dtype,
            **kwargs
        )

        design_mixture_loc = parse_design(
            data=data,
            design_matrix=design_mixture_loc,
            design_key=design_mixture_loc_key,
            dims=INPUT_DATA_PARAMS["design_mixture_loc"]
        )
        if cast_dtype is not None:
            design_mixture_loc = design_mixture_loc.astype(cast_dtype)
            # design = design.chunk({"observations": 1})
        retval.design_mixture_loc = design_mixture_loc

        design_mixture_scale = parse_design(
            data=data,
            design_matrix=design_mixture_scale,
            design_key=design_mixture_scale_key,
            dims=INPUT_DATA_PARAMS["design_mixture_scale"]
        )
        if cast_dtype is not None:
            design_mixture_scale = design_mixture_scale.astype(cast_dtype)
            # design = design.chunk({"observations": 1})
        retval.design_mixture_scale = design_mixture_scale

        if mixture_weight_constraints is not None:
            retval.mixture_weight_constraints = mixture_weight_constraints

        return retval

    @property
    def design_mixture_loc(self) -> xr.DataArray:
        return self.data["design_mixture_loc"]

    @design_mixture_loc.setter
    def design_mixture_loc(self, data):
        self.data["design_mixture_loc"] = data

    @property
    def design_mixture_scale(self) -> xr.DataArray:
        return self.data["design_mixture_scale"]

    @design_mixture_scale.setter
    def design_mixture_scale(self, data):
        self.data["design_mixture_scale"] = data

    @property
    def mixture_weight_constraints(self) -> Union[xr.DataArray, None]:
        return self.data.get("mixture_weight_constraints")

    @mixture_weight_constraints.setter
    def mixture_weight_constraints(self, data):
        if data is None and "mixture_weight_constraints" in self.data:
            del self.data["mixture_weight_constraints"]
        else:
            dims = self.param_shapes()["mixture_weight_constraints"]
            self.data["mixture_weight_constraints"] = xr.DataArray(
                dims=dims,
                data=np.broadcast_to(data, [self.data.dims[d] for d in dims])
            )

            if isinstance(data, pd.DataFrame):
                self.data.coords["mixtures"] = xr.DataArray(
                    dims=("mixtures",),
                    data=np.asarray(data.columns)
                )
                self.data.coords["mixture_group"] = xr.DataArray(
                    dims=("observations",),
                    data=np.asarray(data.index)
                )

    @property
    def num_mixtures(self):
        return self.data.dims["mixtures"]

    @property
    def num_design_mixture_loc_params(self):
        return self.data.dims["design_mixture_loc_params"]

    @property
    def num_design_mixture_scale_params(self):
        return self.data.dims["design_mixture_scale_params"]


class Model(MixtureModel, NB_GLM_Model, metaclass=abc.ABCMeta):
    """
    Generalized Linear Model (GLM) with negative binomial noise.
    Uses the natural logarithm as linker function.
    """

    @classmethod
    def clip_param(cls, param, name):
        return np_clip_param(param, name)

    @classmethod
    def param_shapes(cls) -> dict:
        return MODEL_PARAMS

    @property
    @abc.abstractmethod
    def a(self) -> xr.DataArray:
        pass

    @property
    @abc.abstractmethod
    def b(self) -> xr.DataArray:
        pass

    @property
    @abc.abstractmethod
    def design_mixture_loc(self) -> xr.DataArray:
        pass

    @property
    @abc.abstractmethod
    def design_mixture_scale(self) -> xr.DataArray:
        pass

    # @property
    # def mu(self) -> xr.DataArray:
    #     # exp(design * a + sf)
    #     # Depend on xarray and use xr.DataArray.dot() for matrix multiplication
    #     # Also, make sure that the right order of dimensions get returned => use xr.DataArray.transpose()
    #     log_retval = self.design_loc.dot(self., dims="design_loc_params").transpose(*self.param_shapes()["mu"])
    #
    #     if self.size_factors is not None:
    #         log_retval += self.size_factors
    #
    #     return np.exp(log_retval)
    #
    # @property
    # def r(self) -> xr.DataArray:
    #     # exp(design * b + sf)
    #     # Depend on xarray and use xr.DataArray.dot() for matrix multiplication
    #     # Also, make sure that the right order of dimensions get returned => use xr.DataArray.transpose()
    #     log_retval = self.design_scale.dot(self.b, dims="design_scale_params").transpose(*self.param_shapes()["r"])
    #
    #     if self.size_factors is not None:
    #         log_retval += self.size_factors
    #
    #     return np.exp(log_retval)

    @property
    def par_link_loc(self):
        retval = self.design_mixture_loc.dot(self.a, dims="design_mixture_loc_params")
        retval = retval.transpose(*self.param_shapes()["par_link_loc"])
        return retval

    @property
    def par_link_scale(self):
        retval = self.design_mixture_scale.dot(self.b, dims="design_mixture_scale_params")
        retval = retval.transpose(*self.param_shapes()["par_link_scale"])
        return retval

    def elemwise_log_prob(self) -> xr.DataArray:
        X = self.X
        mu = self.clip_param(self.mu, "mu")
        r = self.clip_param(self.r, "r")
        log_probs = rand_utils.NegativeBinomial(mean=mu, r=r).log_prob(X)
        log_probs: xr.DataArray = self.clip_param(log_probs, "log_probs")
        return log_probs

    def log_probs(self) -> xr.DataArray:
        return self.mixture_log_prob.dot(self.elemwise_log_prob(), dims="mixtures")

    def expected_mixture_prob(self):
        log_probs = self.elemwise_log_prob()

        retval = log_probs.sum(dim="features")
        if self.mixture_weight_constraints is not None:
            with np.errstate(divide='ignore'):
                retval += np.log(self.mixture_weight_constraints).compute()

        retval = softmax(retval, axis=0).transpose(
            *self.param_shapes()["mixture_prob"]
        )

        return retval

    # @property
    # def mu(self) -> xr.DataArray:
    #     return np.exp(np.matmul(self.design_loc, self.a))
    #
    # @property
    # def r(self) -> xr.DataArray:
    #     return np.exp(np.matmul(self.design_loc, self.b))

    def export_params(self, append_to=None, **kwargs):
        if append_to is not None:
            if isinstance(append_to, anndata.AnnData):
                # append_to.obsm["design"] = self.design
                append_to.varm["a"] = np.transpose(self.a)
                append_to.varm["b"] = np.transpose(self.b)
                append_to.obsm["mixture_log_prob"] = self.mixture_log_prob
            elif isinstance(append_to, xr.Dataset):
                # append_to["design"] = (self.param_shapes()["design"], self.design)
                append_to["a"] = (self.param_shapes()["a"], self.a)
                append_to["b"] = (self.param_shapes()["b"], self.b)
                append_to["mixture_log_prob"] = (self.param_shapes()["mixture_log_prob"], self.mixture_log_prob)
            else:
                raise ValueError("Unsupported data type: %s" % str(type(append_to)))
        else:
            ds = xr.Dataset({
                # "design": (self.param_shapes()["design"], self.design),
                "a": (self.param_shapes()["a"], self.a),
                "b": (self.param_shapes()["b"], self.b),
                "mixture_log_prob": (self.param_shapes()["mixture_log_prob"], self.mixture_log_prob),
            })
            return ds


def _model_from_params(
        data: Union[xr.Dataset, anndata.AnnData, xr.DataArray],
        params=None,
        a=None,
        b=None,
        mixture_log_prob=None
):
    input_data = InputData.new(data)

    if params is None:
        if isinstance(data, Model):
            params = xr.Dataset({
                "a": data.a,
                "b": data.b,
                "mixture_log_prob": data.mixture_log_prob,
            })
        elif anndata is not None and isinstance(data, anndata.AnnData):
            params = xr.Dataset({
                "a": (MODEL_PARAMS["a"], np.transpose(data.varm["a"])),
                "b": (MODEL_PARAMS["b"], np.transpose(data.varm["b"])),
                "mixture_log_prob": (MODEL_PARAMS["mixture_log_prob"], data.obsm["mixture_log_prob"]),
            })
        elif isinstance(data, xr.Dataset):
            params = data
        else:
            params = xr.Dataset({
                "a": (MODEL_PARAMS["a"], a) if not isinstance(a, xr.DataArray) else a,
                "b": (MODEL_PARAMS["b"], b) if not isinstance(b, xr.DataArray) else b,
                "mixture_log_prob": (MODEL_PARAMS["mixture_log_prob"], mixture_log_prob) if not isinstance(
                    mixture_log_prob, xr.DataArray) else mixture_log_prob,
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

    @property
    def mixture_log_prob(self):
        return self.params["mixture_log_prob"]

    def __str__(self):
        return "[%s.%s object at %s]: data=%s" % (
            type(self).__module__,
            type(self).__name__,
            hex(id(self)),
            self.params
        )

    def __repr__(self):
        return self.__str__()


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
        params = estim.to_xarray(["a", "b", "mixture_log_prob", "loss", "gradient"], coords=input_data.data)

        XArrayModel.__init__(self, input_data, params)

    @property
    def input_data(self):
        return self._input_data

    @property
    def design_mixture_loc(self) -> xr.DataArray:
        return self.input_data.design_mixture_loc

    @property
    def design_mixture_scale(self) -> xr.DataArray:
        return self.input_data.design_mixture_scale

    @property
    def loss(self):
        return self.params["loss"]

    @property
    def gradient(self):
        return self.params["gradient"]

    # @property
    # def hessians(self):
    #     return self.params["hessians"]
    #
    # @property
    # def fisher_inv(self):
    #     return self.params["fisher_inv"]
