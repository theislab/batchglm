import numpy as np

from .model import Model
from .external import rand_utils, _Simulator_GLM


class Simulator(_Simulator_GLM, Model):
    """
    Simulator for Generalized Linear Models (GLMs) with normal noise.
    Uses the identity as linker function for loc and a log-linker function for scale.
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
            rand_fn_ave=lambda shape: np.random.uniform(1e5, 2 * 1e5, shape),
            rand_fn=lambda shape: np.random.uniform(1.5, 10, shape),
            rand_fn_loc=lambda shape: np.random.uniform(100, 200, shape),
            rand_fn_scale=None,
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
        Sample random data based on normal distribution and parameters.
        """
        self.data["X"] = (
            self.param_shapes()["X"],
            rand_utils.Normal(mean=self.mean, sd=self.sd).sample()
        )
