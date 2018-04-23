import abc

import numpy as np

import utils.random as rand_utils

from models import BasicSimulator

from .base import Model, InputData


class Simulator(BasicSimulator, Model, metaclass=abc.ABCMeta):
    # static variables
    cfg = BasicSimulator.cfg.copy()

    # type hinting
    data: InputData

    def __init__(self, num_samples=2000, num_genes=10000, num_mixtures=2, *args):
        BasicSimulator.__init__(self, *args)

        self.num_genes = num_genes
        self.num_samples = num_samples
        self.num_mixtures = num_mixtures

        self.data = InputData()

    def load(self, folder):
        super().load(folder)

        # if len(self.data.sample_data.shape) < 3:
        #     self.data.sample_data = np.expand_dims(self.data.sample_data, axis=0)

        self.num_genes = self.data.sample_data.shape[-1]
        self.num_samples = self.data.sample_data.shape[-2]
        self.num_mixtures = self.params["sigma2"].shape[-3]


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

    def generate_params(self, *args, min_mean=20, max_mean=10000, min_r=10, max_r=100, prob_transition=0.9, **kwargs):
        dist = rand_utils.NegativeBinomial(
            mean=np.random.uniform(min_mean, max_mean, [self.num_mixtures, 1, self.num_genes]),
            r=np.round(np.random.uniform(min_r, max_r, [self.num_mixtures, 1, self.num_genes]))
        )
        self.params["mu"] = dist.mean
        self.params["sigma2"] = dist.variance

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

        self.data["initial_mixture_probs"] = initial_mixture_probs
        self.params["mixture_assignment"] = real_mixture_assignment
        self.params["mixture_probs"] = real_mixture_probs

    def generate_data(self):
        self.data.sample_data = rand_utils.NegativeBinomial(mean=self.mu, variance=self.sigma2).sample()


def main():
    sim = Simulator()
    sim.generate()
    sim.save("resources/")
