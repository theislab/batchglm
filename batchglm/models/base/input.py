import abc
import os
import logging
from typing import Union

import numpy as np
import scipy
import scipy.sparse
import xarray as xr

try:
    import anndata
except ImportError:
    anndata = None

from .external import pkg_constants, data_utils, SparseXArrayDataSet, SparseXArrayDataArray

logger = logging.getLogger(__name__)

INPUT_DATA_PARAMS = {
    "X": ("observations", "features"),
}

class _InputData_Base:
    """
    Base class for all input data types.
    """
    data: Union[xr.Dataset, SparseXArrayDataSet]

    @classmethod
    @abc.abstractmethod
    def param_shapes(cls) -> dict:
        """
        This method should return a dict of {parameter: (dim0_name, dim1_name, ..)} mappings
        for all parameters of this estimator.
        """
        raise NotImplementedError()

    @classmethod
    def new(cls, data, observation_names=None, feature_names=None, cast_dtype=None):
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
        :param observation_names: (optional) names of the observations.
        :param feature_names: (optional) names of the features.
        :param cast_dtype: data type of all data; should be either float32 or float64
        :return: InputData object
        """
        X = data_utils.xarray_from_data(data)

        if cast_dtype is not None:
            X = X.astype(cast_dtype)
            # X = X.chunk({"observations": 1})

        if scipy.sparse.issparse(X):
            retval = cls(SparseXArrayDataSet(
                X=X,
                obs_names=observation_names,
                feature_names=feature_names
            ))
        elif isinstance(X, SparseXArrayDataArray):
            retval = cls(SparseXArrayDataSet(
                X=X.X,
                obs_names=X.coords[X.dims[0]] if observation_names is None else observation_names,
                feature_names=X.coords[X.dims[1]] if feature_names is None else feature_names,
                dims=X.dims
            ))
        elif isinstance(X, SparseXArrayDataSet):
            retval = cls(X)
            if observation_names is not None:
                retval.observations = observation_names
            if feature_names is not None:
                retval.features = feature_names
        else:
            retval = cls(xr.Dataset({
                "X": X,
            }, coords={
                "feature_allzero": ~X.any(dim="observations")
            }))
            if observation_names is not None:
                retval.observations = observation_names
            elif "observations" not in retval.data.coords:
                retval.observations = retval.data.coords["observations"]

            if feature_names is not None:
                retval.features = feature_names
            elif "features" not in retval.data.coords:
                retval.features = retval.data.coords["features"]

        return retval

    @classmethod
    def from_file(cls, path, group=""):
        """
        Loads pre-sampled data and parameters from specified HDF5 file
        :param path: the path to the HDF5 file
        :param group: the group inside the HDF5 file
        """
        path = os.path.expanduser(path)

        data = xr.open_dataset(
            path,
            group=group,
            engine=pkg_constants.XARRAY_NETCDF_ENGINE
        )

        return cls(data)

    def __init__(self, data):
        self.data = data

    def save(self, path, group="", append=False):
        """
        Saves parameters and sampled data to specified file in HDF5 format
        :param path: the path to the target file where the data will be saved
        :param group: the group inside the HDF5 file where the data will be saved
        :param append: if False, existing files under the specified path will be replaced.
        """
        path = os.path.expanduser(path)
        if os.path.exists(path) and not append:
            os.remove(path)

        mode = "a"
        if not os.path.exists(path):
            mode = "w"

        self.data.to_netcdf(
            path,
            group=group,
            mode=mode,
            engine=pkg_constants.XARRAY_NETCDF_ENGINE
        )

    @property
    def X(self):
        return self.data.X

    @X.setter
    def X(self, data):
        self.data["X"] = data

    @property
    def num_observations(self):
        return self.data.dims["observations"]

    @property
    def num_features(self):
        return self.data.dims["features"]

    @property
    def observations(self):
        return self.data.coords["observations"]

    @observations.setter
    def observations(self, data):
        self.data.coords["observations"] = data

    @property
    def features(self):
        return self.data.coords["features"]

    @features.setter
    def features(self, data):
        self.data.coords["features"] = data

    @property
    def feature_isnonzero(self):
        return ~self.feature_isallzero

    @property
    def feature_isallzero(self):
        return self.data.coords["feature_allzero"]

    def fetch_X_dense(self, idx):
        return self.X[idx].values

    def fetch_X_sparse(self, idx):
        assert isinstance(self.X.X, scipy.sparse.csr_matrix), "tried to fetch sparse from non csr matrix"

        data = self.X.X[idx]

        data_idx = np.asarray(np.vstack(data.nonzero()).T, np.int64)
        data_val = np.asarray(data.data, np.float64)
        data_shape = np.asarray(data.shape, np.int64)

        if idx.shape[0] == 1:
            data_val = np.squeeze(data_val, axis=0)
            data_idx = np.squeeze(data_idx, axis=0)

        return data_idx, data_val, data_shape

    def set_chunk_size(self, cs: int):
        self.X = self.X.chunk({"observations": cs})

    def __copy__(self):
        return type(self)(self.data)

    def __getitem__(self, item):
        if isinstance(item, slice):
            data = self.data.isel(observations=item)
        elif isinstance(item, tuple):
            data = self.data.isel(observations=item[0], features=item[1])
        else:
            data = self.data.isel(observations=item)

        return type(self)(data)

    def __str__(self):
        return "[%s.%s object at %s]: data=%s" % (
            type(self).__module__,
            type(self).__name__,
            hex(id(self)),
            self.data
        )

    def __repr__(self):
        return self.__str__()
