import abc

import math
import numpy as np
import xarray as xr

from .input import _InputData_GLM
from .external import _Simulator_Base


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

    def generate_sample_description(self, num_conditions=2, num_batches=4, **kwargs):
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
    def input_data(self) -> _InputData_GLM:
        return _InputData_GLM.new(self.data)

    @property
    def design_loc(self):
        return self.data["design_loc"]

    @property
    def design_scale(self):
        return self.data["design_scale"]

    @property
    def size_factors(self):
        return self.data.coords.get("size_factors")

    @size_factors.setter
    def size_factors(self, data):
        if data is None and "size_factors" in self.data.coords:
            del self.data.coords["size_factors"]
        else:
            dims = self.param_shapes()["size_factors"]
            self.data.coords["size_factors"] = xr.DataArray(
                dims=dims,
                data=np.broadcast_to(data, [self.data.dims[d] for d in dims])
            )

    @property
    def a(self):
        return self.params['a']

    @property
    def b(self):
        return self.params['b']

