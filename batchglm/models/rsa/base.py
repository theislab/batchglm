import abc
from typing import Union

try:
    import anndata
except ImportError:
    anndata = None
import xarray as xr
import numpy as np

from ..nb_glm.base import InputData as NB_GLM_InputData
from ..nb_glm.base import Model as NB_GLM_Model
from ..nb_glm.base import MODEL_PARAMS as NB_GLM_MODEL_PARAMS
from ..nb_glm.base import INPUT_DATA_PARAMS as NB_GLM_INPUT_DATA_PARAMS
from ..base import BasicEstimator
from ..mixture import Model as MixtureModel

INPUT_DATA_PARAMS = NB_GLM_INPUT_DATA_PARAMS.copy()
INPUT_DATA_PARAMS.update({
    "design_loc": ("observations", "design_loc_params"),
    "design_scale": ("observations", "design_scale_params"),
})

MODEL_PARAMS = NB_GLM_MODEL_PARAMS.copy()
MODEL_PARAMS.update(INPUT_DATA_PARAMS)
MODEL_PARAMS.update({
    "a": ("mixtures", "design_loc_params", "features"),
    "b": ("mixtures", "design_scale_params", "features"),
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
    "hessians": ("features", "delta_var0", "delta_var1"),
    "fisher_inv": ("features", "delta_var0", "delta_var1"),
})


class InputData(NB_GLM_InputData):
    """
    Input data for Generalized Linear Mixture Models (GLMMs) with negative binomial noise.
    """

    @classmethod
    def new(
            cls,
            *args,
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
            *args,
            **kwargs
        )

        return retval


class Model(MixtureModel, NB_GLM_Model, metaclass=abc.ABCMeta):
    """
    Generalized Linear Model (GLM) with negative binomial noise.
    Uses the natural logarithm as linker function.
    """

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
    def mixture_log_prob(self):
        pass

    def log_probs(self):
        self.mixture_log_prob.dot(super().log_probs(), dims="mixtures")

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
        params = estim.to_xarray(["a", "b", "loss", "gradient"], coords=input_data.data)

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

    # @property
    # def hessians(self):
    #     return self.params["hessians"]
    #
    # @property
    # def fisher_inv(self):
    #     return self.params["fisher_inv"]
