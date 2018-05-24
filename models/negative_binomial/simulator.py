import abc

import numpy as np

import utils.random as rand_utils

from models import BasicSimulator

from .base import Model


class Simulator(BasicSimulator, Model, metaclass=abc.ABCMeta):
    
    @property
    def mu(self):
        return np.tile(self.params['mu'], (self.num_samples, 1))
    
    @property
    def sigma2(self):
        return np.tile(self.params['sigma2'], (self.num_samples, 1))
    
    def generate_params(self, *args, min_mean=200, max_mean=100000, min_r=10, max_r=100, **kwargs):
        """
        
        :param min_mean: minimum mean value
        :param max_mean: maximum mean value
        :param min_r: minimum r value
        :param max_r: maximum r value
        """
        dist = rand_utils.NegativeBinomial(
            mean=np.random.uniform(min_mean, max_mean, [self.num_genes]),
            r=np.round(np.random.uniform(min_r, max_r, [self.num_genes]))
        )
        self.params["mu"] = ("genes", dist.mean)
        self.params["sigma2"] = ("genes", dist.variance)
    
    def generate_data(self):
        self.data["sample_data"] = (
            ("samples", "genes"),
            rand_utils.NegativeBinomial(mean=self.mu, variance=self.sigma2).sample()
        )


def sim_test():
    sim = Simulator()
    sim.generate()
    sim.save("test.h5")
    return sim
