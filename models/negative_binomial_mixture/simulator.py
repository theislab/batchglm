import abc

import numpy as np

from utils.random import negative_binomial

from models import BasicSimulator

from .base import Model, InputData


class Simulator(BasicSimulator, Model, metaclass=abc.ABCMeta):
    # static variables
    cfg = BasicSimulator.cfg.copy()
    
    # type hinting
    data: InputData
    
    def __init__(self, num_samples=2000, num_distributions=10000, num_mixtures=2, *args):
        BasicSimulator.__init__(self, *args)
        
        self.num_distributions = num_distributions
        self.num_samples = num_samples
        self.num_mixtures = num_mixtures
        
        self.data = InputData(None)
    
    def load(self, folder):
        super().load(folder)
        
        if len(self.data.sample_data.shape) < 3:
            self.data.sample_data = np.expand_dims(self.data.sample_data, axis=0)
        
        self.num_distributions = self.data.sample_data.shape[-1]
        self.num_samples = self.data.sample_data.shape[-2]
        self.num_mixtures = self.params["r"].shape[-3]
    
    @property
    def r(self):
        retVal = np.tile(self.params['r'], (self.num_samples, 1))
        return retVal[self.mixture_assignment, np.arange(len(self.mixture_assignment))]
    
    @property
    def mu(self):
        retVal = np.tile(self.params['mu'], (self.num_samples, 1))
        return retVal[self.mixture_assignment, np.arange(len(self.mixture_assignment))]
    
    @property
    def p(self):
        # - p*(mu - 1) = p * r / (-p)
        return self.mu / (self.r + self.mu)
    
    @property
    def mixture_assignment(self):
        return self.params["mixture_assignment"]
    
    def generate_params(self, *args, min_mean=20, max_mean=10000, min_r=10, max_r=100, prob_transition=0.9, **kwargs):
        self.params["mu"] = np.random.uniform(min_mean, max_mean, [self.num_mixtures, 1, self.num_distributions])
        self.params["r"] = np.round(np.random.uniform(min_r, max_r, [self.num_mixtures, 1, self.num_distributions]))
        
        initial_mixture_assignment = np.repeat(
            range(self.num_mixtures), np.ceil(self.num_samples / self.num_mixtures)
        )[:self.num_samples]
        
        mixtures = np.random.uniform(0, 1, [self.num_samples])
        mixtures = np.where(mixtures < prob_transition, 1, 0)
        mixtures *= initial_mixture_assignment

        self.data["initial_mixture_assignment"] = initial_mixture_assignment
        self.params["mixture_assignment"] = mixtures
    
    def generate_data(self):
        self.data.sample_data = negative_binomial(self.r, self.mu)


def main():
    sim = Simulator()
    sim.generate()
    sim.save("resources/")
