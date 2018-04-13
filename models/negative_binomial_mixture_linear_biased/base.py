import abc

import numpy as np

from models.negative_binomial_mixture import NegativeBinomialMixtureModel, NegativeBinomialMixtureInputData
from models.negative_binomial_linear_biased import NegativeBinomialWithLinearBiasModel, \
    NegativeBinomialWithLinearBiasInputData

from models import BasicEstimator


class InputData(NegativeBinomialMixtureInputData, NegativeBinomialWithLinearBiasInputData):
    pass


class Model(NegativeBinomialMixtureModel, NegativeBinomialWithLinearBiasInputData, metaclass=abc.ABCMeta):
    pass


class AbstractEstimator(Model, BasicEstimator, metaclass=abc.ABCMeta):
    """
    $\forall i,j,k: \sum_{k}{p_{j,k} * L_{NB}(\mu_{i,k}, \phi_{i,k})}$
    """
    input_data: InputData
    
    def validate_data(self) -> np.ndarray:
        smpls = np.mean(self.input_data.sample_data, 0) < np.var(self.input_data.sample_data, 0)
        
        removed_smpls = np.where(smpls is False)
        print("removing samples due to too small variance: \n%s" % removed_smpls)
        
        self.input_data.sample_data = self.input_data.sample_data[:, np.where(smpls)]
        
        return removed_smpls
