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

from .external import pkg_constants, data_utils

logger = logging.getLogger(__name__)

INPUT_DATA_PARAMS = {
    "X": ("observations", "features"),
}


class SparseXArrayDataArray:
    def __init__(
            self,
            X,
            feature_names,
            obs_names,
            dims=("observations", "features")
    ):
        if feature_names is None:
            feature_names = ["feature_" + str(i) for i in range(X.shape[1])]
        if obs_names is None:
            obs_names = ["obs_" + str(i) for i in range(X.shape[0])]

        self.X = X
        self.dim_names = dims
        self.dims = {
            dims[0]: len(obs_names),
            dims[1]: len(feature_names)
        }
        self.coords = {
            dims[0]: obs_names,
            dims[1]: feature_names,
            "feature_allzero": np.asarray(X.sum(axis=0)).flatten() == 0,
            "size_factors": np.ones([X.shape[0]])
        }

    @classmethod
    def new(
            cls,
            X,
            feature_names,
            obs_names,
            dims
    ):
        retval = cls(
            X=X,
            feature_names=feature_names,
            obs_names=obs_names,
            dims=dims
        )

        return retval

    @property
    def dtype(self):
        return self.X.dtype

    @property
    def ndim(self):
        return len(self.dims)

    def assign_coords(self, coords):
        self.coords.update({coords[0]: coords[1]})

    def square(self):
        self.X = self.X.power(n=2)

    def mean(self):
        return self.X.mean(axis=0)

    def groupby(self, key):
        grouping = self.coords[key]
        groups = np.unique(self.coords[key]).tolist()
        self.groups = np.arange(0, len(groups))
        self.grouping = np.array([groups.index(x) for x in grouping])

    def add(self, a):
        assert a.shape[0] == self.X.shape[0]
        assert a.shape[1] == self.X.shape[1]
        new_x = self.new(
            X=scipy.sparse.csc_matrix(self.X + a),
            feature_names=self.coords[self.dim_names[0]],
            obs_names=self.coords[self.dim_names[1]],
            dims=self.dim_names
        )
        return new_x

    def group_add(self, a):
        assert a.shape[0] == self.X.shape[1]
        assert a.shape[1] == len(self.groups)
        new_x = self.new(
            X=self.X + a[:, self.grouping],
            feature_names=self.coords[self.dim_names[0]],
            obs_names=self.coords[self.dim_names[1]],
            dims=self.dim_names
        )
        return new_x

    def group_means(self):
        group_means = np.vstack([self.X[np.where(self.grouping == x)[0], :].mean(axis=0)
                                 for x in self.groups])
        return group_means

    def group_var(self):
        N = self.X.shape[0]
        Xsq = self.X ** 2
        group_means = np.vstack([self.X[np.where(self.grouping == x)[0], :].mean(axis=0)
                                 for x in self.groups])
        group_var = np.vstack([Xsq[np.where(self.grouping == x)[0], :].multiply(1/N)-group_means[:,i] ** 2
                                 for i,x in enumerate(self.groups)])
        return group_var


class SparseXArrayDataSet:
    """
    Behaves like xarray but carries a scipy.sparse.csr_array.

    Importantly, data.X can be still be efficiently row sliced.
    """

    def __init__(
            self,
            X,
            feature_names,
            obs_names,
            dims=("observations", "features")
    ):
        if feature_names is None:
            feature_names = ["feature_" + str(i) for i in range(X.shape[1])]
        if obs_names is None:
            obs_names = ["obs_" + str(i) for i in range(X.shape[0])]

        self.X = SparseXArrayDataArray(
            X=X,
            feature_names=feature_names,
            obs_names=obs_names,
            dims=dims
        )
        self.dim_names = dims
        self.dims = {
            dims[0]: len(obs_names),
            dims[1]: len(feature_names)
        }
        self.coords = {
            dims[0]: obs_names,
            dims[1]: feature_names,
            "feature_allzero": np.asarray(X.sum(axis=0)).flatten() == 0,
            "size_factors": np.ones([X.shape[0]])
        }

    @property
    def ndim(self):
        return len(self.dims)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        for dim_i in value.dims:
            if dim_i not in self.dim_names:
                self.coords.update({dim_i: value.coords[dim_i]})
                self.dims.update({dim_i: len(value.coords[dim_i])})
            else:
                assert len(value.coords[dim_i]) == len(self.coords[dim_i])
        self.__setattr__(key, value)


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
                    feature_names=feature_names,
                    obs_names=observation_names
            ))
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

    def fetch_X(self, idx):
        return self.X[idx].values

    def fetch_X_sparse(self, idx):
        data = self.X.X[idx]
        data_shape_0 = len(idx)
        # Need to cast also integer columns into data accuracy to that
        # this is a homogeneous tensor in the tensorflow interface.
        data_idx = np.asarray(np.vstack(data.nonzero()).T, dtype=np.int64)
        data_val = data.data

        if idx.size == 1:
            data_val = np.squeeze(data_val, axis=0)
            data_idx = np.squeeze(data_idx, axis=0)

        return data_idx, data_val, data_shape_0

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
