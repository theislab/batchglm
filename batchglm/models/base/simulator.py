import abc
import logging

import dask.array
import numpy as np

try:
    import anndata
except ImportError:
    anndata = None

logger = logging.getLogger(__name__)


class _SimulatorBase(metaclass=abc.ABCMeta):
    """
    Simulator base class.

    Classes implementing `BasicSimulator` should be able to generate a
    2D-matrix of sample data, as well as a dict of corresponding parameters.

    convention: N features with M observations each => (M, N) matrix

    Attributes
    ----------
    nobs : int
        Number of observations
    nfeatures : int
        Number of features

    chunk_size_cells : int
        dask chunk size for cells
    chunk_size_genes : int
        dask chunk size for genes
    """

    nobs: int
    nfeatures: int

    def __init__(self, num_observations: int, num_features: int):
        self.nobs = num_observations
        self.nfeatures = num_features

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
