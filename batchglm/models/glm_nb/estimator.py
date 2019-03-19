import abc

from .model import Model, Model_XArray
from .external import _Estimator_GLM, _EstimatorStore_XArray_GLM, ESTIMATOR_PARAMS


class AbstractEstimator(Model, _Estimator_GLM, metaclass=abc.ABCMeta):
    r"""
    Estimator base class for generalized linear models (GLMs) with
    negative binomial noise.
    """

    @classmethod
    def param_shapes(cls) -> dict:
        return ESTIMATOR_PARAMS


class EstimatorStoreXArray(_EstimatorStore_XArray_GLM, AbstractEstimator, Model_XArray):

    def __init__(self, estim: AbstractEstimator):
        input_data = estim.input_data
        # to_xarray triggers the get function of these properties and thereby
        # causes evaluation of the properties that have not been computed during
        # training, such as the hessian.
        params = estim.to_xarray(
            ["a_var", "b_var", "loss", "log_likelihood", "gradients", "fisher_inv"],
            coords=input_data.data
        )

        Model_XArray.__init__(self, input_data, params)