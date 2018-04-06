import abc

import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import NegativeBinomial as NBdist

from utils.random import negative_binomial

from models import BasicSimulator

from .base import Model, InputData


class Simulator(BasicSimulator, Model, metaclass=abc.ABCMeta):
    # static variables
    cfg = BasicSimulator.cfg.copy()
    
    # type hinting
    data: InputData
    
    def __init__(self, num_samples=2000, num_distributions=10000, *args):
        BasicSimulator.__init__(self, *args)
        
        self.num_samples = num_samples
        self.num_distributions = num_distributions
        
        self.data = InputData(None)
    
    def load(self, folder):
        super().load(folder)
        
        self.num_samples = self.data.sample_data.shape[0]
        self.num_distributions = self.data.sample_data.shape[1]
    
    @property
    def r(self):
        return np.tile(self.params['r'], (self.num_samples, 1))
    
    @property
    def p(self):
        # - p*(mu - 1) = p * r / (-p)
        return self.mu / (self.r + self.mu)
    
    @property
    def mu(self):
        return np.tile(self.params['mu'], (self.num_samples, 1))
    
    def generate_params(self, *args, min_mean=20, max_mean=10000, min_r=10, max_r=100, **kwargs):
        self.params["mu"] = np.random.uniform(min_mean, max_mean, [self.num_distributions])
        self.params["r"] = np.round(np.random.uniform(min_r, max_r, [self.num_distributions]))
    
    def generate_data(self):
        self.data.sample_data = negative_binomial(self.r, self.mu)


def main():
    sim = Simulator()
    sim.generate()
    sim.save("resources/")
