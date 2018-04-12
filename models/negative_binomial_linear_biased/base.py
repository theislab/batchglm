import abc

import numpy as np

from models.negative_binomial import NegativeBinomialModel, NegativeBinomialInputData
from models import BasicEstimator


class InputData(NegativeBinomialInputData):

    def __init__(self, sample_data, design):
        super().__init__(sample_data)
        self.design = design

    @property
    def design(self):
        return self['design']

    @design.setter
    def design(self, value):
        self['design'] = value


class Model(NegativeBinomialModel, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def a(self):
        pass

    @property
    @abc.abstractmethod
    def b(self):
        pass


class AbstractEstimator(Model, BasicEstimator, metaclass=abc.ABCMeta):
    input_data: InputData

    def validate_data(self) -> np.ndarray:
        smpls = np.mean(self.input_data.sample_data, 0) < np.var(self.input_data.sample_data, 0)

        removed_smpls = np.where(smpls == False)
        print("removing samples due to too small variance: \n%s" % removed_smpls)

        self.input_data.sample_data = np.squeeze(self.input_data.sample_data[:, np.where(smpls)])

        return removed_smpls
