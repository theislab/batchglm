import abc

import numpy as np
# import pandas as pd
# import patsy

import data as data_utils
from models.negative_binomial_mixture import NegativeBinomialMixtureSimulator
from models.negative_binomial_linear_biased.simulator import generate_sample_description

from .base import Model


class Simulator(NegativeBinomialMixtureSimulator, Model, metaclass=abc.ABCMeta):
    
    def __init__(self, *args, num_samples=2000, num_genes=10000, num_mixtures=2, **kwargs):
        NegativeBinomialMixtureSimulator.__init__(
            self,
            *args,
            num_samples=num_samples,
            num_genes=num_genes,
            num_mixtures=num_mixtures,
            **kwargs
        )
        Model.__init__(self)
    
    def generate_sample_description(self, num_batches=4, num_confounder=2):
        sample_description = generate_sample_description(self.num_samples, num_batches, num_confounder)
        self.data.merge(sample_description, inplace=True)
    
    def generate_params(self, *args, min_bias=0.5, max_bias=2, **kwargs):
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
        :param min_bias: minimum bias factor of design parameters
        :param max_bias: maximum bias factor of design parameters
        """
        super().generate_params(*args, **kwargs)
        
        if not "sample_description" in self.data:
            self.generate_sample_description()
        
        if not "design" in self.data:
            data_utils.design_matrix_from_dataset(self.data, inplace=True)
        
        self.params['a'] = (
            ("mixtures","design_params", "genes"),
            np.log(np.concatenate([
                self.params["mu"],
                np.random.uniform(
                    min_bias,
                    max_bias,
                    (self.num_mixtures, self.data.design.shape[1] - 1, self.num_genes)
                )
            ], axis=-2))
        )
        
        self.params['b'] = (
            ("mixtures","design_params", "genes"),
            np.log(np.concatenate([
                self.params["sigma2"],
                np.random.uniform(
                    min_bias,
                    max_bias,
                    (self.num_mixtures, self.data.design.shape[1] - 1, self.num_genes)
                )
            ], axis=-2))
        )
    
    @property
    def mu(self):
        retval = np.exp(np.matmul(self.data.design, self.params['a']))
        retval = retval[self.mixture_assignment, np.arange(len(self.mixture_assignment))]
        return retval
    
    @property
    def sigma2(self):
        retval = np.exp(np.matmul(self.data.design, self.params['b']))
        retval = retval[self.mixture_assignment, np.arange(len(self.mixture_assignment))]
        return retval
    
    @property
    def a(self):
        return self.params['a']
    
    @property
    def b(self):
        return self.params['b']


def sim_test():
    sim = Simulator()
    sim.generate()
    sim.save("test.h5")
    return sim
