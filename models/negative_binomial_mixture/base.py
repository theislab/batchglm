import abc

import numpy as np

from models import BasicEstimator
from models import BasicInputData


class InputData(BasicInputData):
    # same as BasicInputData
    @property
    def initial_mixture_probs(self):
        return self['initial_mixture_probs']


class Model(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def r(self):
        pass

    @property
    @abc.abstractmethod
    def p(self):
        pass

    @property
    @abc.abstractmethod
    def mu(self):
        pass

    @property
    @abc.abstractmethod
    def mixture_assignment(self):
        pass


class AbstractEstimator(Model, BasicEstimator, metaclass=abc.ABCMeta):
    """
    $\forall j: \sum_{k}{p_{j,k} * L_{NB}(\mu_{j,k}, \phi_{j,k})}$
    """
    input_data: InputData

    def validate_data(self) -> np.ndarray:
        smpls = np.mean(self.input_data.sample_data, 0) < np.var(self.input_data.sample_data, 0)

        removed_smpls = np.where(smpls is False)
        print("removing samples due to too small variance: \n%s" % removed_smpls)

        self.input_data.sample_data = self.input_data.sample_data[:, np.where(smpls)]

        return removed_smpls
