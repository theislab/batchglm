import abc
import logging

import dask.array
import numpy as np

try:
    import anndata
except ImportError:
    anndata = None

from .input import InputDataBase
from .model import _ModelBase

logger = logging.getLogger(__name__)


class _SimulatorBase(metaclass=abc.ABCMeta):
    r"""
    Simulator base class.

    Classes implementing `BasicSimulator` should be able to generate a
    2D-matrix of sample data, as well as a dict of corresponding parameters.

    convention: N features with M observations each => (M, N) matrix
    """

    nobs: int
    nfeatures: int

    input_data: InputDataBase
    model: _ModelBase

    def __init__(self, model, num_observations, num_features):
        self.nobs = num_observations
        self.nfeatures = num_features

        self.input_data = None
        self.model = model

    def generate(self, sparse: bool = False):
        """
        First generates the parameter set, then observations random data using these parameters.

        :param sparse: Description of parameter `sparse`.
        """
        self.generate_params()
        self.generate_data(sparse=sparse)

    @abc.abstractmethod
    def generate_data(self, *args, **kwargs):
        """
        Should sample random data based on distribution and parameters.

        :param type args: TODO.
        :param type kwargs: TODO.
        """
        pass

    @abc.abstractmethod
    def generate_params(self, *args, **kwargs):
        """
        Should generate all necessary parameters.

        :param type args: TODO.
        :param type kwargs: TODO.
        :return: Description of returned object.
        :rtype: type
        """
        pass

    @property
    def x(self) -> np.ndarray:
        if isinstance(self.input_data.x, dask.array.core.Array):
            return self.input_data.x.compute()
        else:
            return self.input_data.x
