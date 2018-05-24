import abc

import numpy as np

import utils.random as rand_utils

from models import BasicSimulator

from .base import Model


class Simulator(BasicSimulator, Model, metaclass=abc.ABCMeta):
    
    def __init__(self, *args, num_samples=2000, num_genes=10000, num_mixtures=2, **kwargs):
        BasicSimulator.__init__(self, *args, **kwargs)
        
        self.num_genes = num_genes
        self.num_samples = num_samples
        self.num_mixtures = num_mixtures
    
    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)
        
        self.num_mixtures = self.data.dims["mixtures"]
    
    @property
    def mu(self):
        retval = np.tile(self.params['mu'], (self.num_samples, 1))
        retval = retval[self.mixture_assignment, np.arange(len(self.mixture_assignment))]
        return np.squeeze(retval)
    
    @property
    def sigma2(self):
        retval = np.tile(self.params['sigma2'], (self.num_samples, 1))
        retval = retval[self.mixture_assignment, np.arange(len(self.mixture_assignment))]
        return np.squeeze(retval)
    
    @property
    def mixture_assignment(self):
        return self.params["mixture_assignment"]
    
    @property
    def mixture_prob(self):
        return self.params["mixture_probs"]
    
    def generate_params(self, *args, min_mean=20, max_mean=10000, min_r=10, max_r=100, prob_transition=0.9,
                        shuffle_mixture_assignment=False, **kwargs):
        """
        
        :param min_mean: minimum mean value
        :param max_mean: maximum mean value
        :param min_r: minimum r value
        :param max_r: maximum r value
        :param prob_transition: probability for transition from mixture 0 to another mixture.
            
            If 'prob_transition' is a scalar, the same transition probability will be applied to all mixtures.
            
            Per-mixture transition probabilities can be provided by a vector of
            probabilites with length 'num_mixtures'.
        :param shuffle_mixture_assignment: should the mixture assignments be shuffled?
            If false, the samples will be divided into 'num_mixtures' parts and continuously assigned with mixtures.
            I.e. the first part will be
        """
        
        dist = rand_utils.NegativeBinomial(
            mean=np.random.uniform(min_mean, max_mean, [self.num_mixtures, 1, self.num_genes]),
            r=np.round(np.random.uniform(min_r, max_r, [self.num_mixtures, 1, self.num_genes]))
        )
        
        self.params["mu"] = (
            ("mixtures", "one-dim", "genes"),
            dist.mean
        )
        self.params["sigma2"] = (
            ("mixtures", "one-dim", "genes"),
            dist.variance
        )
        
        initial_mixture_assignment = np.repeat(
            range(self.num_mixtures), np.ceil(self.num_samples / self.num_mixtures)
        )[:self.num_samples]
        
        real_mixture_assignment = np.random.uniform(0, 1, [self.num_samples])
        real_mixture_assignment = np.where(real_mixture_assignment < prob_transition, 1, 0)
        # idea: [ 0 0 0 | 1 0 1 | 1 1 0] * [ 0 0 0 | 1 1 1 | 2 2 2] = [ 0 0 0 | 1 0 1 | 2 2 0 ]
        real_mixture_assignment *= initial_mixture_assignment
        
        initial_mixture_probs = np.zeros([self.num_mixtures, self.num_samples])
        initial_mixture_probs[initial_mixture_assignment, range(self.num_samples)] = 1
        real_mixture_probs = np.zeros([self.num_mixtures, self.num_samples])
        real_mixture_probs[real_mixture_assignment, range(self.num_samples)] = 1
        
        self.data["initial_mixture_probs"] = (("mixtures", "samples"), initial_mixture_probs)
        self.params["mixture_assignment"] = (("samples"), real_mixture_assignment)
        self.params["mixture_probs"] = (("mixtures", "samples"), real_mixture_probs)
    
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
