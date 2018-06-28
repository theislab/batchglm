import abc

import math
import numpy as np
import xarray as xr
# import pandas as pd
# import patsy

import data as data_utils
from models.negative_binomial.simulator import Simulator as NegativeBinomialSimulator
from .base import Model


def generate_sample_description(num_samples, num_batches=4, num_conditions=2) -> xr.Dataset:
    ds = {}
    var_list = ["~ 1"]

    ds["intercept"] = ("samples", np.repeat(1, num_samples))
    if num_batches > 0:
        # batch column
        reps_batches = math.ceil(num_samples / num_batches)
        batches = np.repeat(range(num_batches), reps_batches)
        batches = batches[range(num_samples)].astype(str)

        ds["batch"] = ("samples", batches)
        var_list.append("batch")

    if num_conditions > 0:
        # condition column
        reps_conditions = math.ceil(num_samples / num_conditions)
        conditions = np.squeeze(np.tile([np.arange(num_conditions)], reps_conditions))
        conditions = conditions[range(num_samples)].astype(str)

        ds["condition"] = ("samples", conditions)
        var_list.append("condition")

    # build sample description
    sample_description = xr.Dataset(ds, attrs={
        "formula": " + ".join(var_list)
    })
    # sample_description = pd.DataFrame(data=sample_description, dtype="category")

    return sample_description


class Simulator(Model, NegativeBinomialSimulator, metaclass=abc.ABCMeta):

    def __init__(self, *args, **kwargs):
        NegativeBinomialSimulator.__init__(self, *args, **kwargs)
        Model.__init__(self)

    def generate_sample_description(self, num_batches=4, num_conditions=2):
        sample_description = generate_sample_description(self.num_samples, num_batches, num_conditions)
        self.data.merge(sample_description, inplace=True)
        self.data.attrs["formula"] = sample_description.attrs["formula"]

        del self.data["intercept"]

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

        if "formula" not in self.data.attrs:
            self.generate_sample_description()

        if "design" not in self.data:
            data_utils.design_matrix_from_xarray(self.data, dim="samples", append=True)

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

        del self.params["mu"]
        del self.params["r"]

    @property
    def design(self):
        return self.data["design"].values

    @property
    def a(self):
        return self.params['a'].values

    @property
    def b(self):
        return self.params['b'].values


def sim_test():
    sim = Simulator()
    sim.generate()
    sim.save("unit_test.h5")
    return sim
