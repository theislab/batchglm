import numpy as np
import xarray as xr

from .input import InputData
from .model import Model, XArrayModel
from .external import data_utils, rand_utils, _Simulator_GLM


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

    def generate_params(
            self,
            *args,
            rand_fn_ave=lambda shape: np.random.poisson(500, shape)+1,
            rand_fn=lambda shape: np.abs(np.random.uniform(0.5, 2, shape)),
            rand_fn_loc=None,
            rand_fn_scale=None,
            **kwargs):
        """
        
        :param min_mean: minimum mean value
        :param max_mean: maximum mean value
        :param min_r: minimum r value
        :param max_r: maximum r value
        :param rand_fn_ave: function which generates random numbers for intercept.
            Takes one location parameter of intercept distribution across features.
        :param rand_fn: random function taking one argument `shape`.
            default: rand_fn = lambda shape: np.random.uniform(0.5, 2, shape)
        :param rand_fn_loc: random function taking one argument `shape`.
            If not provided, will use `rand_fn` instead.
        :param rand_fn_scale: random function taking one argument `shape`.
            If not provided, will use `rand_fn` instead.
        """
        if rand_fn_loc is None:
            rand_fn_loc = rand_fn
        if rand_fn_scale is None:
            rand_fn_scale = rand_fn

        if "design_loc" not in self.data:
            if "formula_loc" not in self.data.attrs:
                self.generate_sample_description()

            dmat = data_utils.design_matrix_from_xarray(self.data, dim="observations", formula_key="formula_loc")
            dmat_ar = xr.DataArray(dmat, dims=self.param_shapes()["design_loc"])
            dmat_ar.coords["design_loc_params"] = dmat.design_info.column_names
            self.data["design_loc"] = dmat_ar
        if "design_scale" not in self.data:
            if "formula_scale" not in self.data.attrs:
                self.generate_sample_description()

            dmat = data_utils.design_matrix_from_xarray(self.data, dim="observations", formula_key="formula_scale")
            dmat_ar = xr.DataArray(dmat, dims=self.param_shapes()["design_scale"])
            dmat_ar.coords["design_scale_params"] = dmat.design_info.column_names
            self.data["design_scale"] = dmat_ar

        self.params['a'] = xr.DataArray(
            dims=self.param_shapes()["a"],
            data=np.log(
                np.concatenate([
                    np.expand_dims(rand_fn_ave(self.num_features), axis=0),  # intercept
                    rand_fn_loc((self.data.design_loc.shape[1] - 1, self.num_features))
                ], axis=0)
            ),
            coords={"design_loc_params": self.data.design_loc_params}
        )
        self.params['b'] = xr.DataArray(
            dims=self.param_shapes()["b"],
            data=np.log(
                np.concatenate([
                    rand_fn_scale((self.data.design_scale.shape[1], self.num_features))
                ], axis=0)
            ),
            coords={"design_scale_params": self.data.design_scale_params}
        )

    def generate_data(self):
        self.data["X"] = (
            self.param_shapes()["X"],
            rand_utils.NegativeBinomial(mean=self.mu, r=self.r).sample()
        )

    @property
    def input_data(self) -> InputData:
        return InputData.new(self.data)
