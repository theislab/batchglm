from typing import Union

try:
    import anndata
except ImportError:
    anndata = None
import xarray as xr
import numpy as np
import pandas as pd

from .utils import parse_design
from .external import BasicInputData

import patsy

INPUT_DATA_PARAMS = {
    "X": ("observations", "features"),
    "design_loc": ("observations", "design_loc_params"),
    "design_scale": ("observations", "design_scale_params"),
    "size_factors": ("observations",),
}

class InputData_GLM(BasicInputData):
    """
    Input data for Generalized Linear Models (GLMs).
    """
    constraints_loc: Union[None, np.ndarray]
    constraints_scale: Union[None, np.ndarray]

    @classmethod
    def param_shapes(cls) -> dict:
        return INPUT_DATA_PARAMS

    @classmethod
    def new(
            cls,
            data: Union[np.ndarray, anndata.AnnData, xr.DataArray, xr.Dataset],
            design_loc: Union[np.ndarray, pd.DataFrame, patsy.design_info.DesignMatrix, xr.DataArray] = None,
            design_loc_names: Union[list, np.ndarray, xr.DataArray] = None,
            design_scale: Union[np.ndarray, pd.DataFrame, patsy.design_info.DesignMatrix, xr.DataArray] = None,
            design_scale_names: Union[list, np.ndarray, xr.DataArray] = None,
            constraints_loc: np.ndarray = None,
            constraints_scale: np.ndarray = None,
            size_factors=None,
            observation_names=None,
            feature_names=None,
            design_loc_key="design_loc",
            design_scale_key="design_scale",
            cast_dtype=None
    ):
        """
        Create a new InputData object.

        :param data: Some data object.
            Can be either:
                - np.ndarray: NumPy array containing the raw data
                - anndata.AnnData: AnnData object containing the count data and optional the design models
                    stored as data.obsm[design_loc] and data.obsm[design_scale]
                - xr.DataArray: DataArray of shape ("observations", "features") containing the raw data
                - xr.Dataset: Dataset containing the raw data as data["X"] and optional the design models
                    stored as data[design_loc] and data[design_scale]
        :param design_loc: Some matrix format (observations x mean model parameters)
            The location design model. Optional if already specified in `data`
        :param design_loc_names: (optional)
            Names of the design_loc parameters.
            The names might already be included in `design_loc`.
            Will be used to find identical columns in two models.
        :param design_scale: Some matrix format (observations x dispersion model parameters)
            The scale design model. Optional if already specified in `data`
        :param design_scale_names: (optional)
            Names of the design_scale parameters.
            The names might already be included in `design_loc`.
            Will be used to find identical columns in two models.
        :param constraints_loc: np.ndarray (constraints on mean model x mean model parameters)
            Constraints for location model.
            Array with constraints in rows and model parameters in columns.
            Each constraint contains non-zero entries for the a of parameters that 
            has to sum to zero. This constraint is enforced by binding one parameter
            to the negative sum of the other parameters, effectively representing that
            parameter as a function of the other parameters. This dependent
            parameter is indicated by a -1 in this array, the independent parameters
            of that constraint (which may be dependent at an earlier constraint)
            are indicated by a 1.
        :param constraints_scale: np.ndarray (constraints on dispersion model x dispersion model parameters)
            Constraints for scale model.
            Array with constraints in rows and model parameters in columns.
            Each constraint contains non-zero entries for the a of parameters that 
            has to sum to zero. This constraint is enforced by binding one parameter
            to the negative sum of the other parameters, effectively representing that
            parameter as a function of the other parameters. This dependent
            parameter is indicated by a -1 in this array, the independent parameters
            of that constraint (which may be dependent at an earlier constraint)
            are indicated by a 1.
        :param size_factors: np.ndarray (observations)
            Constant scale factors of the mean model in the linker space.
        :param observation_names: (optional)
            Names of the observations.
        :param feature_names: (optional)
            Names of the features.
        :param design_loc_key:
            Where to find `design_loc` if `data` is some anndata.AnnData or xarray.Dataset.
        :param design_scale_key:
            Where to find `design_scale` if `data` is some anndata.AnnData or xarray.Dataset.
        :param cast_dtype:
            If this option is set, all provided data will be casted to this data type.
        :return: InputData object
        """
        retval = super(InputData_GLM, cls).new(
            data=data,
            observation_names=observation_names,
            feature_names=feature_names,
            cast_dtype=cast_dtype
        )

        design_loc = parse_design(
            data=data,
            design_matrix=design_loc,
            coords={"design_loc_params": design_loc_names},
            design_key=design_loc_key,
            dims=INPUT_DATA_PARAMS["design_loc"]
        )
        design_scale = parse_design(
            data=data,
            design_matrix=design_scale,
            coords={"design_scale_params": design_scale_names},
            design_key=design_scale_key,
            dims=INPUT_DATA_PARAMS["design_scale"]
        )

        if cast_dtype is not None:
            design_loc = design_loc.astype(cast_dtype)
            design_scale = design_scale.astype(cast_dtype)

        retval.design_loc = design_loc
        retval.design_scale = design_scale

        retval.constraints_loc = constraints_loc
        retval.constraints_scale = constraints_scale

        if size_factors is not None:
            retval.size_factors = size_factors

        return retval

    @property
    def design_loc(self) -> xr.DataArray:
        return self.data["design_loc"]

    @design_loc.setter
    def design_loc(self, data):
        self.data["design_loc"] = data

    @property
    def design_loc_names(self) -> xr.DataArray:
        return self.data.coords["design_loc_params"]

    @design_loc_names.setter
    def design_loc_names(self, data):
        self.data.coords["design_loc_params"] = data

    @property
    def design_scale(self) -> xr.DataArray:
        return self.data["design_scale"]

    @design_scale.setter
    def design_scale(self, data):
        self.data["design_scale"] = data

    @property
    def design_scale_names(self) -> xr.DataArray:
        return self.data.coords["design_scale_params"]

    @design_scale_names.setter
    def design_scale_names(self, data):
        self.data.coords["design_scale_params"] = data

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
    def num_design_loc_params(self):
        return self.data.dims["design_loc_params"]

    @property
    def num_design_scale_params(self):
        return self.data.dims["design_scale_params"]

    def fetch_design_loc(self, idx):
        return self.design_loc[idx]

    def fetch_design_scale(self, idx):
        return self.design_scale[idx]

    def fetch_size_factors(self, idx):
        return self.size_factors[idx]

    def set_chunk_size(self, cs: int):
        """
        Set the chunk size in number of observations

        :param cs: numer of observations in one chunk
        """
        super().set_chunk_size(cs)
        self.design_loc = self.design_loc.chunk({"observations": cs})
        self.design_scale = self.design_scale.chunk({"observations": cs})

    def __str__(self):
        return "[%s.%s object at %s]: data=%s" % (
            type(self).__module__,
            type(self).__name__,
            hex(id(self)),
            str(self.data).replace("\n", "\n    "),
        )

    def __repr__(self):
        return self.__str__()
