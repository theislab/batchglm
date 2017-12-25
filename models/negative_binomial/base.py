import abc
from typing import Union

try:
    import anndata
except ImportError:
    anndata = None
import xarray as xr

from models import BasicEstimator


class Model(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def mu(self):
        pass

    @property
    @abc.abstractmethod
    def r(self):
        pass

    @property
    @abc.abstractmethod
    def sigma2(self):
        pass

    @property
    @abc.abstractmethod
    def count_probs(self):
        pass

    @property
    @abc.abstractmethod
    def log_count_probs(self):
        pass

    @property
    @abc.abstractmethod
    def log_likelihood(self):
        pass


class AbstractEstimator(Model, BasicEstimator, metaclass=abc.ABCMeta):
    input_data: Union[xr.Dataset, anndata.AnnData]
