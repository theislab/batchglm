import abc

import numpy as np

from models.negative_binomial import NegativeBinomialModel
from models import BasicEstimator


class Model(NegativeBinomialModel, metaclass=abc.ABCMeta):
    
    @property
    @abc.abstractmethod
    def a(self):
        pass
    
    @property
    @abc.abstractmethod
    def b(self):
        pass


def validate_data(input_data):
    axis = -2
    sample_data = input_data.sample_data
    _, partitions = np.unique(input_data.design, axis=-2, return_inverse=True)
    
    smpls = list()
    for i in np.unique(partitions):
        data = sample_data[partitions == i, :]
        smpls.append(np.mean(data, -2) < np.var(data, -2))
    smpls = np.asarray(smpls)
    smpls = np.all(smpls, 0)
    
    removed_smpls = np.where(smpls == False)
    print("genes with to too small variance: \n%s" % removed_smpls)
    
    # input_data.sample_data = np.squeeze(input_data.sample_data[:, np.where(smpls)])
    
    return removed_smpls.size == 0


class AbstractEstimator(Model, BasicEstimator, metaclass=abc.ABCMeta):
    
    def validate_data(self):
        return validate_data(self.input_data)
