import abc

import os

import numpy as np
import pandas as pd
import patsy

from models.negative_binomial_mixture import NegativeBinomialMixtureSimulator
from models.negative_binomial_linear_biased.simulator import generate_sample_description

from .base import Model, InputData


class Simulator(NegativeBinomialMixtureSimulator, Model, metaclass=abc.ABCMeta):
    # static variables
    cfg = NegativeBinomialMixtureSimulator.cfg.copy()

    # type hinting
    data: InputData
    sample_description: pd.DataFrame

    def __init__(self, num_samples=2000, num_genes=10000, num_mixtures=2, *args):
        NegativeBinomialMixtureSimulator.__init__(self, *args)

        self.data = InputData()
        self.sample_description = None

    def generate_sample_description(self, num_batches=4, num_confounder=2):
        self.sample_description = generate_sample_description(self.num_samples, num_batches, num_confounder)

    def generate_params(self, *args, min_bias=0.5, max_bias=2, **kwargs):
        super().generate_params(*args, **kwargs)

        if self.sample_description is None:
            self.generate_sample_description()

        self.formula = "~ 1 "
        for col in self.sample_description.columns:
            self.formula += " + %s" % col

        self.data.design = patsy.dmatrix(self.formula, self.sample_description)

        # num_classes = np.unique(self.data.design, axis=0).shape[0]

        self.params['a'] = np.log(
            np.concatenate([
                self.params["r"],
                np.random.uniform(
                    min_bias,
                    max_bias,
                    (self.num_mixtures, self.data.design.shape[1] - 1, self.num_genes)
                )
            ], axis=-2)
        )
        self.params['b'] = np.log(
            np.concatenate([
                self.params["mu"],
                np.random.uniform(
                    min_bias,
                    max_bias,
                    (self.num_mixtures, self.data.design.shape[1] - 1, self.num_genes)
                )
            ], axis=-2)
        )

    @property
    def r(self):
        retval = np.exp(np.matmul(self.data.design, self.params['a']))
        retval = retval[self.mixture_assignment, np.arange(len(self.mixture_assignment))]
        return retval

    @property
    def mu(self):
        retval = np.exp(np.matmul(self.data.design, self.params['b']))
        retval = retval[self.mixture_assignment, np.arange(len(self.mixture_assignment))]
        return retval

    @property
    def a(self):
        return self.params['a']

    @property
    def b(self):
        return self.params['b']

    def load(self, folder):
        super().load(folder)

        file = os.path.join(folder, "sample_description.tsv")
        if os.path.isfile(file):
            self.sample_description = pd.read_csv(file, sep="\t", dtype="category")

            self.formula = "~ 1 "
            for col in self.sample_description.columns:
                self.formula += " + %s" % col

    def save(self, folder):
        super().save(folder)

        file = os.path.join(folder, "sample_description.tsv")
        self.sample_description.to_csv(file, sep="\t", index=False)


def main():
    sim = Simulator()
    sim.generate()
    sim.save("resources/")
