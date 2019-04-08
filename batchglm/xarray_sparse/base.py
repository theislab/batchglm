import logging

import numpy as np
import scipy
import scipy.sparse

logger = logging.getLogger(__name__)

class SparseXArrayDataArray:
    def __init__(
            self,
            X,
            obs_names=None,
            feature_names=None,
            dims=("observations", "features")
    ):
        if obs_names is None:
            obs_names = ["obs_" + str(i) for i in range(X.shape[0])]
        if feature_names is None:
            feature_names = ["feature_" + str(i) for i in range(X.shape[1])]

        self.X = X
        self.dims = dims
        self.coords = {
            dims[0]: obs_names,
            dims[1]: feature_names,
            "feature_allzero": np.asarray(X.sum(axis=0)).flatten() == 0,
            "size_factors": np.ones([X.shape[0]])
        }
        self.groups = None
        self.grouping = None
        self._group_means = None

    @classmethod
    def new(
            cls,
            X,
            obs_names=None,
            feature_names=None,
            dims=None
    ):
        retval = cls(
            X=X,
            obs_names=obs_names,
            feature_names=feature_names,
            dims=dims
        )

        return retval

    def new_from_x(self, x):
        new_x = self.new(
            X=x,
            obs_names=self.coords["observations"],
            feature_names=self.coords["features"],
            dims=self.dims
        )
        new_x.coords = self.coords
        return new_x

    @property
    def dtype(self):
        return self.X.dtype

    def astype(self, dtype):
        return self.new_from_x(self.X.astype(dtype))

    @property
    def shape(self):
        return self.X.shape

    @property
    def ndim(self):
        return len(self.dims)

    @property
    def feature_allzero(self):
        return self.coords["feature_allzero"]

    @feature_allzero.setter
    def feature_allzero(self, data):
        self.coords["feature_allzero"] = data

    @property
    def size_factors(self):
        return self.coords["size_factors"]

    @size_factors.setter
    def size_factors(self, data):
        self.coords["size_factors"] = data

    def assign_coords(self, coords):
        self.coords.update({coords[0]: coords[1]})

    def add(self, a, copy=False):
        assert a.shape[0] == self.X.shape[0]
        assert a.shape[1] == self.X.shape[1]
        assert a.shape[0] == self.X.shape[0]
        assert a.shape[1] == self.X.shape[1]
        new_x = scipy.sparse.csr_matrix(self.X + a)
        if copy:
            return self.new_from_x(new_x)
        else:
            self.X = new_x

    def multiply(self, a, copy=False):
        assert a.shape[0] == self.X.shape[0]
        assert a.shape[1] == self.X.shape[1]
        new_x = self.X.multiply(a).tocsr()
        if copy:
            return self.new_from_x(new_x)
        else:
            self.X = new_x

    def square(self, copy=False):
        new_x = self.X.power(n=2)
        if copy:
            return self.new_from_x(new_x)
        else:
            self.X = new_x

    def mean(self, dim: str = None, axis: int = None):
        assert not (dim is not None and axis is not None), "only supply dim or axis"
        if dim is not None:
            assert dim in self.dims, "dim not recognized"
            axis = self.dims.index(dim)
            return np.asarray(self.X.mean(axis=axis)).flatten()
        elif axis is not None:
            assert axis < len(self.X.shape), "axis index out of range"
            return np.asarray(self.X.mean(axis=axis)).flatten()
        else:
            return np.asarray(self.X.mean()).flatten()

    def var(self, dim: str = None, axis: int = None):
        assert not (dim is not None and axis is not None), "only supply dim or axis"
        if dim is not None:
            assert dim in self.dims, "dim not recognized"
            axis = self.dims.index(dim)
        elif axis is not None:
            assert axis < len(self.X.shape), "axis index out of range"
        else:
            assert False, "supply either dim or axis"

        Xsq = self.square(copy=True)
        expect_x_sq = np.square(self.mean(axis=axis))
        expect_xsq = Xsq.mean(axis=axis)
        return np.asarray(expect_xsq - expect_x_sq).flatten()

    def std(self, dim: str):
        return np.sqrt(self.var(dim=dim))

    def groupby(self, key):
        groups, group_order, grouping = np.unique(self.coords[key], return_index=True, return_inverse=True)
        self.groups = np.arange(0, len(groups))
        self.grouping = grouping  #group_order[grouping]
        self._group_means = None  # Set back to None in case grouping are applied iteratively.

    def group_add(self, a, copy=False):
        assert a.shape[0] == self.X.shape[1]
        assert a.shape[1] == len(self.groups)
        new_x = scipy.sparse.csr_matrix(self.X + a[:, self.grouping])
        if copy:
            return self.new_from_x(new_x)
        else:
            self.X = new_x

    def group_means(self, dim):
        assert dim in self.dims, "dim not recognized"
        assert self.groups is not None, "set groups first"
        axis = self.dims.index(dim)
        if self._group_means is None:
            self._group_means = np.asarray(np.vstack([self.X[np.where(self.grouping == x)[0], :].mean(axis=axis)
                                                      for x in self.groups]))
        return self._group_means

    def group_vars(self, dim):
        assert dim in self.dims, "dim not recognized"
        assert self.groups is not None, "set groups first"
        axis = self.dims.index(dim)
        Xsq = self.square(copy=True)
        expect_xsq = np.vstack([Xsq.X[np.where(self.grouping == x)[0], :].mean(axis=axis)
                                for x in self.groups])
        expect_x_sq = np.square(self.group_means(dim=dim))
        group_vars = np.asarray(expect_xsq - expect_x_sq)
        return group_vars

    def __copy__(self):
        return type(self)(self.X)

    def __getitem__(self, key):
        if isinstance(key, np.ndarray) or isinstance(key, slice):  # This is an observation wise slice!
            return self.new_from_x(x=self.X[key])
        elif isinstance(key, tuple) and len(key) == 2:
            if isinstance(key[0], slice) and isinstance(key[1], np.int):  # This is an gene-wise wise slice!
                # Note: just returning values here and not new instance of class.
                return np.asarray(self.X[:, key[1]].todense()).flatten()
        elif key in self.coords:
            return self.coords[key]
        else:
            return self.__getattribute__(key)


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

    @property
    def shape(self):
        return self.X.shape

    @property
    def feature_allzero(self):
        return self.X.coords["feature_allzero"]

    @feature_allzero.setter
    def feature_allzero(self, data):
        self.coords["feature_allzero"] = data
        self.X.coords["feature_allzero"] = data

    @property
    def size_factors(self):
        return self.X.coords["size_factors"]

    @size_factors.setter
    def size_factors(self, data):
        self.coords["size_factors"] = data
        self.X.coords["size_factors"] = data

    def __getitem__(self, key):
        if key in self.coords:
            return self.coords[key]
        else:
            return self.__getattribute__(key)

    def __setitem__(self, key, value):
        for dim_i in value.dims:
            if dim_i not in self.dim_names:
                self.coords.update({dim_i: value.coords[dim_i]})
                self.dims.update({dim_i: len(value.coords[dim_i])})
            else:
                assert len(value.coords[dim_i]) == len(self.coords[dim_i])
        self.__setattr__(key, value)
