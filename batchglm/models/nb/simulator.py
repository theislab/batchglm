import abc

import numpy as np

from ..external import rand_utils
from ..base import BasicSimulator

from .base import Model, InputData


class Simulator(Model, BasicSimulator, metaclass=abc.ABCMeta):
    """
    Simulator for negative binomial data
    """

    @property
    def input_data(self) -> InputData:
        return InputData.new(self.data)

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
    def X(self):
        return self.data["X"]

    @property
    def mu(self):
        # return np.tile(self.params['mu'], (self.num_observations, 1))
        retval = self.params['mu'].expand_dims("observations")
        retval = retval.isel(observations=np.repeat(0, self.num_observations))
        return retval

    @property
    def r(self):
        # return np.tile(self.params['r'], (self.num_observations, 1))
        retval = self.params['r'].expand_dims("observations")
        retval = retval.isel(observations=np.repeat(0, self.num_observations))
        return retval

    def generate_params(self, *args, min_mean=1, max_mean=10000, min_r=1, max_r=10, **kwargs):
        """
        
        :param min_mean: minimum mean value
        :param max_mean: maximum mean value
        :param min_r: minimum r value
        :param max_r: maximum r value
        """

        mean = np.random.uniform(min_mean, max_mean, [self.num_features])
        r = np.round(np.random.uniform(min_r, max_r, [self.num_features]))

        # dist = rand_utils.NegativeBinomial(
        #     mean=mean,
        #     r=r
        # )

        self.params["mu"] = ("features", mean)
        self.params["r"] = ("features", r)

    def generate_data(self):
        self.data["X"] = (
            self.param_shapes()["X"],
            rand_utils.NegativeBinomial(mean=self.mu, r=self.r).sample()
        )


def sim_test():
    sim = Simulator()
    sim.generate()
    sim.save("unit_test.h5")
    return sim
