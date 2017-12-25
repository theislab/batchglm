import abc

import math
import numpy as np
import xarray as xr
# import pandas as pd
# import patsy

import data as data_utils
from models.negative_binomial import NegativeBinomialSimulator
from .base import Model


def generate_sample_description(num_samples, num_batches=4, num_confounder=2) -> xr.Dataset:
    reps_batches = math.ceil(num_samples / num_batches)
    reps_confounder = math.ceil(num_samples / num_confounder)

    # batch column
    batches = np.repeat(range(num_batches), reps_batches)
    batches = batches[range(num_samples)].astype(str)

    # confounder column
    confounders = np.squeeze(np.tile([np.arange(num_confounder)], reps_confounder))
    confounders = confounders[range(num_samples)].astype(str)

    # build sample description
    sample_description = xr.Dataset({
        "batch": ("samples", batches),
        "confounder": ("samples", confounders),
    }, attrs={
        "formula": "~ 1 + batch + confounder"
    })
    # sample_description = pd.DataFrame(data=sample_description, dtype="category")

    return sample_description


class Simulator(NegativeBinomialSimulator, Model, metaclass=abc.ABCMeta):

    def __init__(self, *args, **kwargs):
        NegativeBinomialSimulator.__init__(self, *args, **kwargs)
        Model.__init__(self)

    def generate_sample_description(self, num_batches=4, num_confounder=2):
        sample_description = generate_sample_description(self.num_samples, num_batches, num_confounder)
        self.data.merge(sample_description, inplace=True)
        self.data.attrs["formula"] = sample_description.attrs["formula"]

    def generate_params(self, *args, min_bias=0.5, max_bias=2, **kwargs):
        """
        
        :param min_mean: minimum mean value
        :param max_mean: maximum mean value
        :param min_r: minimum r value
        :param max_r: maximum r value
        :param min_bias: minimum bias factor of design parameters
        :param max_bias: maximum bias factor of design parameters
        """
        super().generate_params(*args, **kwargs)

        if "sample_description" not in self.data:
            self.generate_sample_description()

        if "design" not in self.data:
            data_utils.design_matrix_from_dataset(self.data, inplace=True)

        self.params['a'] = (
            ("design_params", "genes"),
            np.log(
                np.concatenate([
                    np.expand_dims(self.params["mu"], 0),
                    np.random.uniform(min_bias, max_bias, (self.data.design.shape[1] - 1, self.num_genes))
                ])
            )
        )
        self.params['b'] = (
            ("design_params", "genes"),
            np.log(
                np.concatenate([
                    np.expand_dims(self.params["r"], 0),
                    np.random.uniform(min_bias, max_bias, (self.data.design.shape[1] - 1, self.num_genes))
                ])
            )
        )

    @property
    def mu(self):
        return np.exp(np.matmul(self.data.design, self.params['a']))

    @property
    def r(self):
        return np.exp(np.matmul(self.data.design, self.params['b']))

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
