import logging

import numpy as np
import scipy
import scipy.sparse

logger = logging.getLogger(__name__)

class SparseXArrayDataArray:
    def __init__(
            self,
            X,
            obs_names,
            feature_names,
            dims=("observations", "features")
    ):
        if feature_names is None:
            feature_names = ["feature_" + str(i) for i in range(X.shape[1])]
        if obs_names is None:
            obs_names = ["obs_" + str(i) for i in range(X.shape[0])]

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
    def shape(self):
        return self.X.shape

    @property
    def ndim(self):
        return len(self.dims)

    def assign_coords(self, coords):
        self.coords.update({coords[0]: coords[1]})

    def add(self, a):
        assert a.shape[0] == self.X.shape[0]
        assert a.shape[1] == self.X.shape[1]
        new_x = self.new(
            X=scipy.sparse.csc_matrix(self.X + a),
            obs_names=self.coords["observations"],
            feature_names=self.coords["features"],
            dims=self.dims
        )
        return new_x

    def square(self, copy=False):
        if copy:
            return self.X.power(n=2)
        else:
            self.X = self.X.power(n=2)

    def mean(self, dim=None):
        if dim is not None:
            assert dim in self.dims, "dim not recognized"
            axis = self.dims.index(dim)
            return np.asarray(self.X.mean(axis=axis)).flatten()
        else:
            return np.asarray(self.X.mean()).flatten()

    def var(self, dim):
        assert dim in self.dims, "dim not recognized"
        axis = self.dims.index(dim)
        Xsq = self.square(copy=True)
        expect_x_sq = np.square(self.mean(dim=dim))
        expect_xsq = np.mean(Xsq, axis=axis)
        return np.asarray(expect_xsq - expect_x_sq).flatten()

    def std(self, dim):
        return np.sqrt(self.var(dim=dim))

    def groupby(self, key):
        groups, self.grouping = np.unique(self.coords[key], return_inverse=True)
        self.groups = np.arange(0, len(groups))
        self._group_means = None  # Set back to None in case grouping are applied iteratively.

    def group_add(self, a):
        assert a.shape[0] == self.X.shape[1]
        assert a.shape[1] == len(self.groups)
        new_x = self.new(
            X=self.X + a[:, self.grouping],
            obs_names=self.coords["observations"],
            feature_names=self.coords["features"],
            dims=self.dims
        )
        return new_x

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
        expect_xsq = np.vstack([Xsq[np.where(self.grouping == x)[0], :].mean(axis=axis)
                                for x in self.groups])
        expect_x_sq = np.square(self.group_means(dim=dim))
        group_vars = np.asarray(expect_xsq - expect_x_sq)
        return group_vars


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
