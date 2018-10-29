import abc

import numpy as np
import xarray as xr
# import pandas as pd
# import patsy

from ..nb_glm.simulator import Simulator as NB_GLM_Simulator
from ..external import data_utils, rand_utils
from .base import Model, InputData


class Simulator(Model, NB_GLM_Simulator, metaclass=abc.ABCMeta):
    """
    Simulator for Generalized Linear Models (GLMs) with negative binomial noise.
    Uses the natural logarithm as linker function.
    """

    def __init__(self, *args, num_mixtures=2, **kwargs):
        NB_GLM_Simulator.__init__(self, *args, **kwargs)
        Model.__init__(self)

        self.num_mixtures = num_mixtures

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

    @property
    def num_mixtures(self):
        return self._num_mixtures

    @num_mixtures.setter
    def num_mixtures(self, data):
        self._num_mixtures = data

    def generate_params(self, *args, min_mean=1, max_mean=10000, min_r=1, max_r=10,
                        rand_fn=lambda shape: np.random.uniform(0.5, 2, shape),
                        rand_fn_loc=None,
                        rand_fn_scale=None,
                        prob_transition=0.9,
                        shuffle_mixture_assignment=False, min_bias=0.5, max_bias=2, **kwargs):
        r"""

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
        :param prob_transition: probability for transition from mixture 0 to another mixture.

            If 'prob_transition' is a scalar, the same transition probability will be applied to all mixtures.

            Per-mixture transition probabilities can be provided by a vector of
            probabilites with length 'num_mixtures'.
        :param shuffle_mixture_assignment: should the mixture assignments be shuffled?
            If false, the observations will be divided into 'num_mixtures' parts and continuously assigned with mixtures.
            I.e. the first part will be
        :param min_bias: minimum bias factor of design parameters
        :param max_bias: maximum bias factor of design parameters
        """
        mean = np.random.uniform(min_mean, max_mean, [self.num_mixtures, 1, self.num_features])
        r = np.random.uniform(min_r, max_r, [self.num_mixtures, 1, self.num_features])

        # self.params["mu"] = (
        #     ("mixtures", "one-dim", "features"),
        #     mean
        # )
        # self.params["r"] = (
        #     ("mixtures", "one-dim", "features"),
        #     r
        # )

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
            data=np.log(np.concatenate([
                mean,
                rand_fn_loc((self.num_mixtures, self.data.design_loc.shape[1] - 1, self.num_features))
            ], axis=-2)),
            coords={"design_loc_params": self.data.design_loc_params}
        )

        self.params['b'] = xr.DataArray(
            dims=self.param_shapes()["b"],
            data=np.log(np.concatenate([
                r,
                rand_fn_scale((self.num_mixtures, self.data.design_scale.shape[1] - 1, self.num_features))
            ], axis=-2)),
            coords={"design_scale_params": self.data.design_scale_params}
        )

        initial_mixture_assignment = np.repeat(
            range(self.num_mixtures), np.ceil(self.num_observations / self.num_mixtures)
        )[:self.num_observations]

        real_mixture_assignment = np.random.uniform(0, 1, [self.num_observations])
        real_mixture_assignment = np.where(real_mixture_assignment < prob_transition, 1, 0)
        # idea: [ 0 0 0 | 1 0 1 | 1 1 0] * [ 0 0 0 | 1 1 1 | 2 2 2] = [ 0 0 0 | 1 0 1 | 2 2 0 ]
        real_mixture_assignment *= initial_mixture_assignment

        initial_mixture_probs = np.tile(np.nextafter(0, 1, dtype=float), [self.num_mixtures, self.num_observations])
        initial_mixture_probs[initial_mixture_assignment, range(self.num_observations)] = 1
        real_mixture_probs = np.tile(np.nextafter(0, 1, dtype=float), [self.num_mixtures, self.num_observations])
        real_mixture_probs[real_mixture_assignment, range(self.num_observations)] = 1

        self.data["initial_mixture_probs"] = xr.DataArray(
            dims=self.param_shapes()["mixture_log_prob"],
            data=np.transpose(initial_mixture_probs)
        )
        self.params["mixture_assignment"] = xr.DataArray(
            dims=self.param_shapes()["mixture_assignment"],
            data=real_mixture_assignment
        )
        self.params["mixture_log_prob"] = xr.DataArray(
            dims=self.param_shapes()["mixture_log_prob"],
            data=np.transpose(np.log(real_mixture_probs))
        )

    def generate_data(self):
        self.data["X"] = (
            self.param_shapes()["X"],
            rand_utils.NegativeBinomial(
                mean=self.mu[self.mixture_assignment, range(self.num_observations)],
                r=self.r[self.mixture_assignment, range(self.num_observations)],
            ).sample()
        )

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)

        self.num_mixtures = self.data.dims["mixtures"]

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

    @property
    def mixture_log_prob(self):
        return self.params["mixture_log_prob"]


def sim_test():
    sim = Simulator()
    sim.generate()
    sim.save("unit_test.h5")
    return sim
