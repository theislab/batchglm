import abc

import numpy as np

from models import BasicEstimator
from models import BasicInputData


class InputData(BasicInputData):
    # same as BasicInputData
    pass


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


class AbstractEstimator(Model, BasicEstimator, metaclass=abc.ABCMeta):
    input_data: InputData

    def validate_data(self) -> np.ndarray:
        smpls = np.mean(self.input_data.sample_data, 0) < np.var(self.input_data.sample_data, 0)

        removed_smpls = np.where(smpls is False)
        print("removing samples due to too small variance: \n%s" % removed_smpls)

        self.input_data.sample_data = np.squeeze(self.input_data.sample_data[:, np.where(smpls)])

        return removed_smpls
