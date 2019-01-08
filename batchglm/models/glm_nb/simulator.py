from .model import Model
from .external import rand_utils, _Simulator_GLM


class Simulator(_Simulator_GLM, Model):
    """
    Simulator for Generalized Linear Models (GLMs) with negative binomial noise.
    Uses the natural logarithm as linker function.
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

    def generate_data(self):
        """
        Sample random data based on negative binomial distribution and parameters.
        """
        self.data["X"] = (
            self.param_shapes()["X"],
            rand_utils.NegativeBinomial(mean=self.mu, r=self.r).sample()
        )
