import abc

try:
    import anndata
except ImportError:
    anndata = None

from .model import Model, XArrayModel, MODEL_PARAMS
from .external import BasicEstimator

ESTIMATOR_PARAMS = MODEL_PARAMS.copy()
ESTIMATOR_PARAMS.update({
    "loss": (),
    "gradient": ("features",),
    "hessians": ("features", "delta_var0", "delta_var1"),
    "fisher_inv": ("features", "delta_var0", "delta_var1"),
})

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
        # to_xarray triggers the get function of these properties and thereby
        # causes evaluation of the properties that have not been computed during
        # training, such as the hessian.
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
        return self.params["gradient"]

    @property
    def hessians(self):
        return self.params["hessians"]

    @property
    def fisher_inv(self):
        return self.params["fisher_inv"]
