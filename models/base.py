import abc

import numpy as np
import os


class BasicInputData(dict):

    @property
    def sample_data(self):
        return self['sample_data']

    @sample_data.setter
    def sample_data(self, value):
        self['sample_data'] = value


class BasicEstimator(metaclass=abc.ABCMeta):
    input_data: dict
    loss: any

    def __init__(self, input_data: dict):
        self.input_data = input_data

    @abc.abstractmethod
    def validate_data(self, **kwargs):
        pass

    @abc.abstractmethod
    def initialize(self, **kwargs):
        pass

    @abc.abstractmethod
    def train(self, **kwargs):
        pass


class BasicSimulator(metaclass=abc.ABCMeta):
    data: BasicInputData
    params: dict

    """
    Classes implementing `MatrixSimulator` should be able to generate a
    2D-matrix of sample data, as well as a dict of corresponding parameters.

    convention: N genes with M samples each => (M, N) matrix
    """
    cfg = {
        "data_folder": "data",
        "param_folder": "params",
    }

    def __init__(self, num_samples=2000, num_genes=10000):
        self.num_samples = num_samples
        self.num_genes = num_genes

        self.data = {}
        self.params = {}

    def generate(self):
        """
        First generates the parameter set, then samples random data using these parameters
        """
        self.generate_params()
        self.generate_data()

    @abc.abstractmethod
    def generate_data(self, *args, **kwargs):
        """
        Should sample random data using the pre-defined / sampled parameters
        """
        pass

    @abc.abstractmethod
    def generate_params(self, *args, **kwargs):
        """
        Should generate all necessary parameters
        """
        pass

    def load(self, folder):
        """
        Loads pre-sampled data and parameters from specified folder
        :param folder: the source folder
        """

        data_folder = os.path.join(folder, self.cfg['data_folder'])
        if os.path.isdir(data_folder):
            for data_name in os.listdir(data_folder):
                file = os.path.join(data_folder, data_name)
                # print(file)
                if os.path.isfile(file):
                    self.data[data_name.replace(".npy", "")] = np.load(file)

        param_folder = os.path.join(folder, self.cfg['param_folder'])
        if os.path.isdir(param_folder):
            for param_name in os.listdir(param_folder):
                file = os.path.join(param_folder, param_name)
                # print(file)
                if os.path.isfile(file):
                    self.params[param_name.replace(".npy", "")] = np.load(file)

    def save(self, folder):
        """
        Saves parameters and sampled data to specified folder
        :param folder: the target folder where the data will be saved
        """
        data_folder = os.path.join(folder, self.cfg['data_folder'])
        os.makedirs(data_folder, exist_ok=True)

        for (data, val) in self.data.items():
            # print("saving param '%s' to %s" % param, os.path.join(param_folder, param))
            np.save(os.path.join(data_folder, data), val)

        param_folder = os.path.join(folder, self.cfg['param_folder'])
        os.makedirs(param_folder, exist_ok=True)

        for (param, val) in self.params.items():
            # print("saving param '%s' to %s" % param, os.path.join(param_folder, param))
            np.save(os.path.join(param_folder, param), val)
