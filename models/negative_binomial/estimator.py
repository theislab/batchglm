import numpy as np

from models import BasicEstimator

from . import NegativeBinomialModel, NegativeBinomialInputData

__all__ = ['AbstractNegativeBinomialEstimator']


class AbstractNegativeBinomialEstimator(NegativeBinomialModel, BasicEstimator):
    input_data: NegativeBinomialInputData
    
    def validateData(self) -> np.ndarray:
        smpls = np.mean(self.input_data.sample_data, 0) < np.var(self.input_data.sample_data, 0)
        
        removed_smpls = np.where(smpls == False)
        print("removing samples due to too small variance: \n%s" % removed_smpls)
        
        self.input_data.sample_data = self.input_data.sample_data[:, np.where(smpls)]
        
        return removed_smpls

