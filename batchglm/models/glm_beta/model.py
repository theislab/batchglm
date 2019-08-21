import abc
try:
    import anndata
except ImportError:
    anndata = None
import numpy as np

from .external import _ModelGLM


class Model(_ModelGLM, metaclass=abc.ABCMeta):
    """
    Generalized Linear Model (GLM) with beta distributed noise, logit link for location and log link for scale.
    """

    def link_loc(self, data):
        return np.log(1/(1/data-1))

    def inverse_link_loc(self, data):
        return 1/(1+np.exp(-data))

    def link_scale(self, data):
        return np.log(data)

    def inverse_link_scale(self, data):
        return np.exp(data)

    @property
    def eta_loc(self) -> np.ndarray:
        eta = np.matmul(self.design_loc, self.a)
        if self.size_factors is not None:
            assert False, "size factors not allowed"
        return eta

    # Re-parameterizations:

    @property
    def mean(self) -> np.ndarray:
        return self.location

    @property
    def samplesize(self) -> np.ndarray:
        return self.scale

    @property
    def p(self) -> np.ndarray:
        return self.mean * self.samplesize

    @property
    def q(self) -> np.ndarray:
        return (1 - self.mean) * self.samplesize
