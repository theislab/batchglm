import abc

import numpy as np

from models.negative_binomial import NegativeBinomialModel, NegativeBinomialInputData
from models import BasicEstimator


class InputData(NegativeBinomialInputData):
    
    @property
    def initial_mixture_probs(self):
        return self['initial_mixture_probs']


class Model(NegativeBinomialModel, metaclass=abc.ABCMeta):
    
    @property
    @abc.abstractmethod
    def mixture_assignment(self):
        pass
    
    @property
    @abc.abstractmethod
    def mixture_prob(self):
        pass


def validate_data(input_data) -> np.ndarray:
    smpls = np.mean(input_data.sample_data, 0) < np.var(input_data.sample_data, 0)
    
    removed_smpls = np.where(smpls == False)
    print("removing samples due to too small variance: \n%s" % removed_smpls)
    
    input_data.sample_data = np.squeeze(input_data.sample_data[:, np.where(smpls)])
    
    return input_data


class AbstractEstimator(Model, BasicEstimator, metaclass=abc.ABCMeta):
    """
    $\forall i,j,k: \sum_{k}{p_{j,k} * L_{NB}(\mu_{i,k}, \phi_{i,k})}$
    """
    input_data: InputData
    
    def validate_data(self):
        self.input_data = validate_data(self.input_data)
