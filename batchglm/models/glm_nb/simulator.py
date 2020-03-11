import numpy as np

from .model import Model
from .external import _SimulatorGLM, InputDataGLM
from .external import pkg_constants


class Simulator(_SimulatorGLM, Model):
    """
    Simulator for Generalized Linear Models (GLMs) with negative binomial noise.
    Uses the natural logarithm as linker function.
    """

    def __init__(
            self,
            num_observations=1000,
            num_features=100
    ):
        Model.__init__(
            self=self,
            input_data=None
        )
        _SimulatorGLM.__init__(
            self=self,
            model=None,
            num_observations=num_observations,
            num_features=num_features
        )

    def generate_params(
            self,
            rand_fn_ave=lambda shape: np.random.poisson(500, shape) + 1,
            rand_fn=lambda shape: np.abs(np.random.uniform(0.5, 2, shape)),
            rand_fn_loc=None,
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
        Sample random data based on negative binomial distribution and parameters.
        """
        data_matrix = np.random.negative_binomial(
            n=self.phi,
            p=1 - self.mu / (self.phi + self.mu),
            size=None
        )
        self.input_data = InputDataGLM(
            data=data_matrix,
            design_loc=self.sim_design_loc,
            design_scale=self.sim_design_scale,
            design_loc_names=None,
            design_scale_names=None
        )

    def param_bounds(
            self,
            dtype
    ):
        # TODO: inherit this from somewhere?
        dtype = np.dtype(dtype)
        dmin = np.finfo(dtype).min
        dmax = np.finfo(dtype).max
        dtype = dtype.type

        sf = dtype(pkg_constants.ACCURACY_MARGIN_RELATIVE_TO_LIMIT)
        bounds_min = {
            "a_var": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "b_var": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "eta_loc": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "eta_scale": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "loc": np.nextafter(0, np.inf, dtype=dtype),
            "scale": np.nextafter(0, np.inf, dtype=dtype),
            "likelihood": dtype(0),
            "ll": np.log(np.nextafter(0, np.inf, dtype=dtype)),
        }
        bounds_max = {
            "a_var": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "b_var": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "eta_loc": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "eta_scale": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "loc": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "scale": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "likelihood": dtype(1),
            "ll": dtype(0),
        }
        return bounds_min, bounds_max
