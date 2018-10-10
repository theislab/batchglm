import abc

import math
import numpy as np
import xarray as xr
# import pandas as pd
# import patsy

from ..nb.simulator import Simulator as NegativeBinomialSimulator
from ..external import data_utils
from .base import Model, InputData


def generate_sample_description(num_observations, num_conditions=2, num_batches=4) -> xr.Dataset:
    """ Build a design matrix.

    :param num_observations: Number of cells to simulate.
    :param num_batchs
    """
    ds = {}
    var_list = ["~ 1"]

    ds["intercept"] = ("observations", np.repeat(1, num_observations))
    if num_conditions > 0:
        # condition column
        reps_conditions = math.ceil(num_observations / num_conditions)
        conditions = np.squeeze(np.tile([np.arange(num_conditions)], reps_conditions))
        conditions = conditions[range(num_observations)].astype(str)

        ds["condition"] = ("observations", conditions)
        var_list.append("condition")

    if num_batches > 0:
        # batch column
        reps_batches = math.ceil(num_observations / num_batches)
        batches = np.repeat(range(num_batches), reps_batches)
        batches = batches[range(num_observations)].astype(str)

        ds["batch"] = ("observations", batches)
        var_list.append("batch")

    # build sample description
    sample_description = xr.Dataset(ds, attrs={
        "formula": " + ".join(var_list)
    })
    # sample_description = pd.DataFrame(data=sample_description, dtype="category")

    return sample_description


class Simulator(Model, NegativeBinomialSimulator, metaclass=abc.ABCMeta):
    """
    Simulator for Generalized Linear Models (GLMs) with negative binomial noise.
    Uses the natural logarithm as linker function.
    """

    def __init__(self, *args, **kwargs):
        NegativeBinomialSimulator.__init__(self, *args, **kwargs)
        Model.__init__(self)

    @property
    def num_observations(self):
        return self._num_observations

    @num_observations.setter
    def num_observations(self, data):
        self._num_observations = data

    @property
    def num_features(self):
        return self._num_features

    @num_features.setter
    def num_features(self, data):
        self._num_features = data

    def generate_sample_description(self, num_conditions=2, num_batches=4):
        sample_description = generate_sample_description(
            self.num_observations, 
            num_conditions=num_conditions,
            num_batches=num_batches
        )
        self.data.merge(sample_description, inplace=True)
        self.data.attrs["formula"] = sample_description.attrs["formula"]

        del self.data["intercept"]

    def parse_dmat_loc(self, dmat):
        """ Input externally created design matrix for location model.
        """
        self.data.attrs["formula"] = None
        dmat_ar = xr.DataArray(dmat, dims=self.param_shapes()["design_loc"])
        dmat_ar.coords["design_loc_params"] = ["p"+str(i) for i in range(dmat.shape[1])]
        self.data["design_loc"] = dmat_ar
        
    def parse_dmat_scale(self, dmat):
        """ Input externally created design matrix for scale model.
        """
        self.data.attrs["formula"] = None
        dmat_ar = xr.DataArray(dmat, dims=self.param_shapes()["design_scale"])
        dmat_ar.coords["design_scale_params"] = ["p"+str(i) for i in range(dmat.shape[1])]
        self.data["design_scale"] = dmat_ar
        
    def generate_params(
            self,
            *args,
            rand_fn=lambda shape: np.random.uniform(0.5, 2, shape),
            rand_fn_loc=None,
            rand_fn_scale=None,
            **kwargs):
        """
        
        :param min_mean: minimum mean value
        :param max_mean: maximum mean value
        :param min_r: minimum r value
        :param max_r: maximum r value
        :param rand_fn: random function taking one argument `shape`.
            default: rand_fn = lambda shape: np.random.uniform(0.5, 2, shape)
        :param rand_fn_loc: random function taking one argument `shape`.
            If not provided, will use `rand_fn` instead.
        :param rand_fn_scale: random function taking one argument `shape`.
            If not provided, will use `rand_fn` instead.
        """
        super().generate_params(*args, **kwargs)

        if rand_fn_loc is None:
            rand_fn_loc = rand_fn
        if rand_fn_scale is None:
            rand_fn_scale = rand_fn

        if "formula" not in self.data.attrs:
            self.generate_sample_description()

        if "design_loc" not in self.data:
            dmat = data_utils.design_matrix_from_xarray(self.data, dim="observations")
            dmat_ar = xr.DataArray(dmat, dims=self.param_shapes()["design_loc"])
            dmat_ar.coords["design_loc_params"] = dmat.design_info.column_names
            self.data["design_loc"] = dmat_ar
        if "design_scale" not in self.data:
            dmat = data_utils.design_matrix_from_xarray(self.data, dim="observations")
            dmat_ar = xr.DataArray(dmat, dims=self.param_shapes()["design_scale"])
            dmat_ar.coords["design_scale_params"] = dmat.design_info.column_names
            self.data["design_scale"] = dmat_ar

        self.params['a'] = xr.DataArray(
            dims=self.param_shapes()["a"],
            data=np.log(
                np.concatenate([
                    np.expand_dims(self.params["mu"], 0),
                    rand_fn_loc((self.data.design_loc.shape[1] - 1, self.num_features))
                ])
            ),
            coords={"design_loc_params": self.data.design_loc_params}
        )
        self.params['b'] = xr.DataArray(
            dims=self.param_shapes()["b"],
            data=np.log(
                np.concatenate([
                    np.expand_dims(self.params["r"], 0),
                    rand_fn_scale((self.data.design_scale.shape[1] - 1, self.num_features))
                ])
            ),
            coords={"design_scale_params": self.data.design_scale_params}
        )

        del self.params["mu"]
        del self.params["r"]

    @property
    def input_data(self) -> InputData:
        return InputData.new(self.data)

    @property
    def X(self):
        return self.data["X"]

    @property
    def design_loc(self):
        return self.data["design_loc"]

    @property
    def design_scale(self):
        return self.data["design_scale"]

    @property
    def size_factors(self):
        return None

    @property
    def a(self):
        return self.params['a']

    @property
    def b(self):
        return self.params['b']


def sim_test():
    sim = Simulator()
    sim.generate()
    sim.save("unit_test.h5")
    return sim
