from typing import Union

try:
    import anndata
except ImportError:
    anndata = None
import xarray as xr
import numpy as np
import scipy.sparse
import pandas as pd

from .utils import parse_constraints, parse_design
from .external import _InputData_Base, INPUT_DATA_PARAMS, SparseXArrayDataSet

import patsy

INPUT_DATA_PARAMS = INPUT_DATA_PARAMS
INPUT_DATA_PARAMS.update({
    "design_loc": ("observations", "design_loc_params"),
    "design_scale": ("observations", "design_scale_params"),
    "constraints_loc": ("design_loc_params", "loc_params"),
    "constraints_scale": ("design_scale_params", "scale_params"),
    "size_factors": ("observations",),
})

class InputData(_InputData_Base):
    """
    Input data for Generalized Linear Models (GLMs).
    """
    @classmethod
    def param_shapes(cls) -> dict:
        return INPUT_DATA_PARAMS

    @classmethod
    def new(
            cls,
            data: Union[np.ndarray, anndata.AnnData, xr.DataArray, xr.Dataset, scipy.sparse.csr_matrix],
            design_loc: Union[np.ndarray, pd.DataFrame, patsy.design_info.DesignMatrix, xr.DataArray] = None,
            design_loc_names: Union[list, np.ndarray, xr.DataArray] = None,
            design_scale: Union[np.ndarray, pd.DataFrame, patsy.design_info.DesignMatrix, xr.DataArray] = None,
            design_scale_names: Union[list, np.ndarray, xr.DataArray] = None,
            constraints_loc: Union[np.ndarray, xr.DataArray] = None,
            constraints_scale: Union[np.ndarray, xr.DataArray] = None,
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
        :param constraints_loc: tensor (all parameters x dependent parameters)
            Tensor that encodes how complete parameter set which includes dependent
            parameters arises from indepedent parameters: all = <constraints, indep>.
            This tensor describes this relation for the mean model.
            This form of constraints is used in vector generalized linear models (VGLMs).
        :param constraints_scale: tensor (all parameters x dependent parameters)
            Tensor that encodes how complete parameter set which includes dependent
            parameters arises from indepedent parameters: all = <constraints, indep>.
            This tensor describes this relation for the dispersion model.
            This form of constraints is used in vector generalized linear models (VGLMs).
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
        retval = super(InputData, cls).new(
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

        constraints_loc = parse_constraints(
            dmat=design_loc,
            constraints=constraints_loc,
            dims=INPUT_DATA_PARAMS["constraints_loc"],
            constraint_par_names=None
        )
        constraints_scale = parse_constraints(
            dmat=design_scale,
            constraints=constraints_scale,
            dims=INPUT_DATA_PARAMS["constraints_scale"],
            constraint_par_names=None
        )

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
    def constraints_loc(self) -> xr.DataArray:
        return self.data["constraints_loc"]

    @constraints_loc.setter
    def constraints_loc(self, data):
        self.data["constraints_loc"] = data

    @property
    def loc_names(self) -> xr.DataArray:
        return self.data.coords["loc_params"]

    @loc_names.setter
    def loc_names(self, data):
        self.data.coords["loc_names"] = data

    @property
    def constraints_scale(self) -> xr.DataArray:
        return self.data["constraints_scale"]

    @constraints_scale.setter
    def constraints_scale(self, data):
        self.data["constraints_scale"] = data

    @property
    def scale_names(self) -> xr.DataArray:
        return self.data.coords["scale_params"]

    @scale_names.setter
    def scale_names(self, data):
        self.data.coords["scale_params"] = data

    @property
    def size_factors(self):
        if isinstance(self.data, SparseXArrayDataSet):
            return self.data.size_factors
        else:
            return self.data.coords.get("size_factors")
        #return self.data.coords.get("size_factors")

    @size_factors.setter
    def size_factors(self, data):
        if data is None and "size_factors" in self.data.coords:
            del self.data.coords["size_factors"]

        dims = self.param_shapes()["size_factors"]
        sf = xr.DataArray(
                dims=dims,
                data=np.broadcast_to(data, [self.data.dims[d] for d in dims])
            )
        if isinstance(self.data, SparseXArrayDataSet):
            self.data.size_factors = sf
        else:
            self.data.coords["size_factors"] = sf

    @property
    def num_design_loc_params(self):
        return self.data.dims["design_loc_params"]

    @property
    def num_design_scale_params(self):
        return self.data.dims["design_scale_params"]

    @property
    def num_loc_params(self):
        return self.data.dims["loc_params"]

    @property
    def num_scale_params(self):
        return self.data.dims["scale_params"]

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
