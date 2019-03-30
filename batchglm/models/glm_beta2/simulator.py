import numpy as np

from .model import Model
from .external import rand_utils, _Simulator_GLM


class Simulator(_Simulator_GLM, Model):
    """
    Simulator for Generalized Linear Models (GLMs) with beta2 distributed noise.
    Uses a logit-linker function for loc and a log-linker function for scale.
    """

    def __init__(
            self,
            num_observations=1000,
            num_features=100
    ):
        Model.__init__(self)
        _Simulator_GLM.__init__(
            self,
            num_observations=num_observations,
            num_features=num_features
        )

    def generate_params(
            self,
            rand_fn_ave=lambda shape: np.random.uniform(0.2, 0.8, shape),
            rand_fn=None,
            rand_fn_loc=lambda shape: np.random.uniform(0.5, 0.6, shape),
            rand_fn_scale=lambda shape: np.random.uniform(1e1, 2*1e1, shape),
        ):
        self._generate_params(
            self,
            rand_fn_ave=rand_fn_ave,
            rand_fn=rand_fn,
            rand_fn_loc=rand_fn_loc,
            rand_fn_scale=rand_fn_scale,
        )

    def generate_data(self):
        """
        Sample random data based on beta2 distribution and parameters.
        """
        self.data["X"] = (
            self.param_shapes()["X"],
            rand_utils.Beta2(mean=self.mean, samplesize=self.samplesize).sample()
        )
