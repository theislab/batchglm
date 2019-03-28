import abc
import math
import numpy as np
import xarray as xr

from .input import InputData
from .external import _Simulator_Base, data_utils


def generate_sample_description(
        num_observations,
        num_conditions=2,
        num_batches=4,
        shuffle_assignments=False
) -> xr.Dataset:
    """ Build a sample description.

    :param num_observations: Number of observations to simulate.
    :param num_conditions: number of conditions; will be repeated like [1,2,3,1,2,3]
    :param num_batches: number of conditions; will be repeated like [1,1,2,2,3,3]
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

    if shuffle_assignments:
        sample_description = sample_description.isel(
            observations=np.random.permutation(sample_description.observations.values)
        )

    return sample_description


class _Simulator_GLM(_Simulator_Base, metaclass=abc.ABCMeta):
    """
    Simulator for Generalized Linear Models (GLMs).
    """

    def __init__(
            self,
            num_observations,
            num_features
    ):
        _Simulator_Base.__init__(
            self,
            num_observations=num_observations,
            num_features=num_features
        )

    def generate_sample_description(
            self,
            num_conditions=2,
            num_batches=4,
            **kwargs
    ):
        sample_description = generate_sample_description(
            self.num_observations,
            num_conditions=num_conditions,
            num_batches=num_batches,
            **kwargs
        )
        self.data = self.data.merge(sample_description)

        if "formula_loc" not in self.data.attrs:
            self.data.attrs["formula_loc"] = sample_description.attrs["formula"]
        if "formula_scale" not in self.data.attrs:
            self.data.attrs["formula_scale"] = sample_description.attrs["formula"]

        del self.data["intercept"]

    def _generate_params(
            self,
            *args,
            rand_fn_ave=None,
            rand_fn=None,
            rand_fn_loc=None,
            rand_fn_scale=None,
            **kwargs
    ):
        """
        Generate all necessary parameters

        :param rand_fn_ave: function which generates random numbers for intercept.
            Takes one location parameter of intercept distribution across features.
        :param rand_fn: random function taking one argument `shape`.
        :param rand_fn_loc: random function taking one argument `shape`.
            If not provided, will use `rand_fn` instead.
            This function generates location model parameters in inverse linker space,
            ie. these parameter will be log transformed if a log linker function is used!
            Values below 1e-08 will be set to 1e-08 to map them into the positive support.
        :param rand_fn_scale: random function taking one argument `shape`.
            If not provided, will use `rand_fn` instead.
            This function generates scale model parameters in inverse linker space,
            ie. these parameter will be log transformed if a log linker function is used!
            Values below 1e-08 will be set to 1e-08 to map them into the positive support.
        """
        if rand_fn_ave is None:
            raise ValueError("rand_fn_ave must not be None!")
        if rand_fn is None and rand_fn_loc is None:
            raise ValueError("rand_fn and rand_fn_loc must not be both None!")
        if rand_fn is None and rand_fn_scale is None:
            raise ValueError("rand_fn and rand_fn_scale must not be both None!")

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

        if "constraints_loc" not in self.data:
            constr_loc = np.identity(n=self.data["design_loc"].shape[1])
            constr_loc_ar = xr.DataArray(
                dims=[self.param_shapes()["design_loc"][1], self.param_shapes()["a_var"][0]],
                data=constr_loc,
                coords={"loc_params": self.data.coords["design_loc_params"].values}
            )
            constr_loc_ar.coords["loc_params"] = self.data.coords["design_loc_params"].values
            self.data["constraints_loc"] = constr_loc_ar
        if "constraints_scale" not in self.data:
            constr_scale = np.identity(n=self.data["design_scale"].shape[1])
            constr_scale_ar = xr.DataArray(
                dims=[self.param_shapes()["design_scale"][1], self.param_shapes()["b_var"][0]],
                data=constr_scale,
                coords={"scale_params": self.data.coords["design_scale_params"].values}
            )
            constr_scale_ar.coords["scale_params"] = self.data.coords["design_scale_params"].values
            self.data["constraints_scale"] = constr_scale_ar

        self.params["a_var"] = xr.DataArray(
            dims=self.param_shapes()["a_var"],
            data=self.link_loc(
                np.concatenate([
                    np.expand_dims(rand_fn_ave([self.num_features]), axis=0),  # intercept
                    np.maximum(rand_fn_loc((self.data.design_loc.shape[1] - 1, self.num_features)),
                               np.zeros([self.data.design_loc.shape[1] - 1, self.num_features]) + 1e-08)
                ], axis=0)
            ),
            coords={"loc_params": self.data.loc_params}
        )
        self.params["b_var"] = xr.DataArray(
            dims=self.param_shapes()["b_var"],
            data=self.link_scale(
                np.concatenate([
                    np.maximum(rand_fn_scale((self.data.design_scale.shape[1], self.num_features)),
                               np.zeros([self.data.design_scale.shape[1], self.num_features]) + 1e-08)
                ], axis=0)
            ),
            coords={"scale_params": self.data.scale_params}
        )

    def parse_dmat_loc(self, dmat):
        """ Input externally created design matrix for location model.
        """
        self.data.attrs["formula_loc"] = None
        dmat_ar = xr.DataArray(dmat, dims=self.param_shapes()["design_loc"])
        dmat_ar.coords["design_loc_params"] = ["p" + str(i) for i in range(dmat.shape[1])]
        self.data["design_loc"] = dmat_ar

    def parse_dmat_scale(self, dmat):
        """ Input externally created design matrix for scale model.
        """
        self.data.attrs["formula_scale"] = None
        dmat_ar = xr.DataArray(dmat, dims=self.param_shapes()["design_scale"])
        dmat_ar.coords["design_scale_params"] = ["p" + str(i) for i in range(dmat.shape[1])]
        self.data["design_scale"] = dmat_ar

    @property
    def input_data(self) -> InputData:
        return InputData.new(self.data)

    @property
    def design_loc(self):
        return self.data["design_loc"]

    @property
    def design_scale(self):
        return self.data["design_scale"]

    @property
    def constraints_loc(self):
        return self.data["constraints_loc"]

    @property
    def constraints_scale(self):
        return self.data["constraints_scale"]

    @property
    def size_factors(self):
        return self.data.coords.get("size_factors")

    @size_factors.setter
    def size_factors(self, data):
        if data is None and "size_factors" in self.data.coords:
            del self.data.coords["size_factors"]

        dims = self.param_shapes()["size_factors"]
        self.data.coords["size_factors"] = xr.DataArray(
            dims=dims,
            data=np.broadcast_to(data, [self.data.dims[d] for d in dims])
        )

    @property
    def a_var(self):
        return self.params["a_var"]

    @property
    def b_var(self):
        return self.params["b_var"]

    @property
    def a(self) -> xr.DataArray:
        return self.constraints_loc.dot(self.a_var, dims="loc_params")

    @property
    def b(self) -> xr.DataArray:
        return self.constraints_scale.dot(self.b_var, dims="scale_params")
