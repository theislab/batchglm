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


def validate_data(input_data) -> np.ndarray:
    smpls = np.mean(input_data.sample_data, 0) < np.var(input_data.sample_data, 0)
    
    removed_smpls = np.where(smpls == False)
    print("removing samples due to too small variance: \n%s" % removed_smpls)
    
    input_data.sample_data = np.squeeze(input_data.sample_data[:, np.where(smpls)])
    
    return input_data


class AbstractEstimator(Model, BasicEstimator, metaclass=abc.ABCMeta):
    input_data: InputData
    
    def validate_data(self):
        self.input_data = validate_data(self.input_data)
