import abc

import numpy as np
import xarray as xr

from models import BasicEstimator


class Model(metaclass=abc.ABCMeta):
    
    @property
    @abc.abstractmethod
    def mu(self):
        pass
    
    @property
    @abc.abstractmethod
    def sigma2(self):
        pass


def validate_data(input_data) -> bool:
    smpls = np.mean(input_data.sample_data, 0) < np.var(input_data.sample_data, 0)
    
    removed_smpls = np.where(smpls == False)
    print("genes with to too small variance: \n%s" % removed_smpls)
    
    # input_data.sample_data = np.squeeze(input_data.sample_data[:, np.where(smpls)])
    
    return removed_smpls.size == 0


class AbstractEstimator(Model, BasicEstimator, metaclass=abc.ABCMeta):
    input_data: xr.Dataset
    
    def validate_data(self):
        return validate_data(self.input_data)
