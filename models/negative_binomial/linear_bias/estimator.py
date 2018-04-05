import abc

import numpy as np

from models import BasicEstimator

from .base import Model
from .base import InputData


class AbstractEstimator(Model, BasicEstimator, metaclass=abc.ABCMeta):
    input_data: InputData
    
    def validate_data(self) -> np.ndarray:
        smpls = np.mean(self.input_data.sample_data, 0) < np.var(self.input_data.sample_data, 0)
        
        removed_smpls = np.where(smpls == False)
        print("removing samples due to too small variance: \n%s" % removed_smpls)
        
        self.input_data.sample_data = self.input_data.sample_data[:, np.where(smpls)]
        
        return removed_smpls
