import numpy as np

from models import BasicEstimator

from .base import Model
from .base import InputData

__all__ = ['AbstractEstimator']


class AbstractEstimator(Model, BasicEstimator):
    input_data: InputData
    
    def validateData(self) -> np.ndarray:
        smpls = np.mean(self.input_data.sample_data, 0) < np.var(self.input_data.sample_data, 0)
        
        removed_smpls = np.where(smpls == False)
        print("removing samples due to too small variance: \n%s" % removed_smpls)
        
        self.input_data.sample_data = self.input_data.sample_data[:, np.where(smpls)]
        
        return removed_smpls
