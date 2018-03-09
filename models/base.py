import numpy as np
import os

__all__ = ['BasicInputData', 'BasicEstimator', 'BasicSimulator']


class BasicInputData(dict):
    
    def __init__(self, sample_data):
        super().__init__()
        self.sample_data = sample_data
    
    @property
    def sample_data(self):
        return self['sample_data']
    
    @sample_data.setter
    def sample_data(self, value):
        self['sample_data'] = value


class BasicEstimator:
    input_data: dict
    loss: any
    
    def __init__(self, input_data: dict):
        self.input_data = input_data
    
    @classmethod
    def initialize(self, **kwargs):
        raise NotImplementedError
    
    @classmethod
    def train(self, **kwargs):
        raise NotImplementedError


class BasicSimulator:
    data: BasicInputData
    params: dict
    
    """
    Classes implementing `MatrixSimulator` should be able to generate a
    2D-matrix of sample data, as well as a dict of corresponding parameters.

    convention: N distributions with M samples each => (M, N) matrix
    """
    cfg = {
        "data_folder": "data",
        "param_folder": "params",
    }
    
    def __init__(self, num_samples=2000, num_distributions=10000):
        self.num_samples = num_samples
        self.num_distributions = num_distributions
        
        self.data = {}
        self.params = {}
    
    def generate(self):
        """
        First generates the parameter set, then samples random data using these parameters
        """
        self.generate_params()
        self.generate_data()
    
    @classmethod
    def generate_data(self, *args):
        """
        Should sample random data using the pre-defined / sampled parameters
        """
        raise NotImplementedError
    
    @classmethod
    def generate_params(self, *args):
        """
        Should generate all necessary parameters
        """
        raise NotImplementedError
    
    def load(self, folder):
        """
        Loads pre-sampled data and parameters from specified folder
        :param folder: the source folder
        """
        
        data_folder = os.path.join(folder, self.cfg['data_folder'])
        if os.path.isdir(data_folder):
            for data_name in os.listdir(data_folder):
                file = os.path.join(data_folder, data_name)
                if os.path.isfile(file):
                    self.data[data_name] = np.loadtxt(
                        os.path.join(folder, self.cfg["data"]), delimiter="\t")
        
        param_folder = os.path.join(folder, self.cfg['param_folder'])
        if os.path.isdir(param_folder):
            for param_name in os.listdir(param_folder):
                file = os.path.join(param_folder, param_name)
                if os.path.isfile(file):
                    self.params[param_name] = np.loadtxt(
                        os.path.join(folder, self.cfg["data"]), delimiter="\t")
    
    def save(self, folder):
        """
        Saves parameters and sampled data to specified folder
        :param folder: the target folder where the data will be saved
        """
        data_folder = os.path.join(folder, self.cfg['data_folder'])
        os.makedirs(data_folder, exist_ok=True)
        
        for (data, val) in self.data.items():
            # print("saving param '%s' to %s" % param, os.path.join(param_folder, param))
            np.savetxt(os.path.join(data_folder, data), val, delimiter="\t")
        
        param_folder = os.path.join(folder, self.cfg['param_folder'])
        os.makedirs(param_folder, exist_ok=True)
        
        for (param, val) in self.params.items():
            # print("saving param '%s' to %s" % param, os.path.join(param_folder, param))
            np.savetxt(os.path.join(param_folder, param), val, delimiter="\t")
