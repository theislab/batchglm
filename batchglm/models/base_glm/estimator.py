import abc

try:
    import anndata
except ImportError:
    anndata = None

from .model import MODEL_PARAMS
from .external import _Estimator_Base, _EstimatorStore_XArray_Base

ESTIMATOR_PARAMS = MODEL_PARAMS.copy()
ESTIMATOR_PARAMS.update({
    "loss": (),
    "log_likelihood": ("features",),
    "gradients": ("features",),
    "hessians": ("features", "delta_var0", "delta_var1"),
    "fisher_inv": ("features", "delta_var0", "delta_var1"),
})

class _Estimator_GLM(_Estimator_Base, metaclass=abc.ABCMeta):
    r"""
    Estimator base class for generalized linear models (GLMs).
    """

class _EstimatorStore_XArray_GLM(_EstimatorStore_XArray_Base):

    def __init__(self):
        super(_EstimatorStore_XArray_Base, self).__init__()

    @property
    def log_likelihood(self):
        return self.params["log_likelihood"]

    @property
    def gradients(self):
        return self.params["gradients"]

    @property
    def hessians(self):
        return self.params["hessians"]

    @property
    def fisher_inv(self):
        return self.params["fisher_inv"]
