import abc
import logging
import numpy as np

from .input import InputDataBase
from .model import _ModelBase
from .external import maybe_compute

logger = logging.getLogger(__name__)


class _SimulatorBase(metaclass=abc.ABCMeta):
    r"""
    Simulator base class

    Classes implementing `BasicSimulator` should be able to generate a
    2D-matrix of sample data, as well as a dict of corresponding parameters.

    convention: N features with M observations each => (M, N) matrix
    """
    # TODO: why?
    nobs: int
    nfeatures: int

    input_data: InputDataBase
    model: _ModelBase

    def __init__(
            self,
            model: _ModelBase,
            num_observations: int,
            num_features: int
    ):
        self.nobs = num_observations
        self.nfeatures = num_features

        self.input_data = None
        self.model = model

    def generate(self) -> None:
        """
        First generates the parameter set, then observations random data using these parameters
        """
        self.generate_params()
        self.generate_data()

    @abc.abstractmethod
    def generate_data(self, *args, **kwargs):
        """
        Should sample random data based on distribution and parameters.
        """
        pass

    @abc.abstractmethod
    def generate_params(self, *args, **kwargs):
        """
        Should generate all necessary parameters.
        """
        pass

    # TODO: computed property (self.input_data.x should not change)?
    @property
    def x(self) -> np.ndarray:
        return maybe_compute(self.input_data.x)
