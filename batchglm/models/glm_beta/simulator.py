import numpy as np

from .model import Model
from .external import InputDataGLM, _SimulatorGLM


class Simulator(_SimulatorGLM, Model):
    """
    Simulator for Generalized Linear Models (GLMs) with beta distributed noise.
    Uses a logit-linker function for loc and a log-linker function for scale.
    """

    def __init__(
            self,
            num_observations=1000,
            num_features=100
    ):
        _SimulatorGLM.__init__(
            self=self,
            model=None,
            num_observations=num_observations,
            num_features=num_features
        )

    def generate_params(
            self,
            rand_fn_ave=lambda shape: np.random.uniform(0.2, 0.8, shape),
            rand_fn=None,
            rand_fn_loc=lambda shape: np.random.uniform(0.05, 0.15, shape),
            rand_fn_scale=lambda shape: np.random.uniform(0.2, 0.5, shape),
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
        Sample random data based on beta distribution and parameters.
        """
        data_matrix = np.random.beta(
            a=self.p,
            b=self.q,
            size=None
        )
        self.input_data = InputDataGLM(
            data=data_matrix,
            design_loc=self.sim_design_loc,
            design_scale=self.sim_design_scale,
            design_loc_names=None,
            design_scale_names=None
        )
