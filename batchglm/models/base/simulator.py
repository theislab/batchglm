import abc
import os
import logging

import xarray as xr

try:
    import anndata
except ImportError:
    anndata = None

from .external import pkg_constants

logger = logging.getLogger(__name__)


class _Simulator_Base(metaclass=abc.ABCMeta):
    r"""
    Simulator base class
    """
    _num_observations: int
    _num_features: int

    data: xr.Dataset
    params: xr.Dataset

    """
    Classes implementing `BasicSimulator` should be able to generate a
    2D-matrix of sample data, as well as a dict of corresponding parameters.

    convention: N features with M observations each => (M, N) matrix
    """

    def __init__(self, num_observations, num_features):
        self.num_observations = num_observations
        self.num_features = num_features

        self.data = xr.Dataset()
        self.params = xr.Dataset()

    def generate(self):
        """
        First generates the parameter set, then observations random data using these parameters
        """
        self.generate_params()
        self.generate_data()

    @property
    def X(self):
        return self.data["X"]

    @property
    def num_observations(self):
        return self._num_observations

    @num_observations.setter
    def num_observations(self, data):
        self._num_observations = data

    @property
    def num_features(self):
        return self._num_features

    @num_features.setter
    def num_features(self, data):
        self._num_features = data

    @property
    def sample_description(self):
        return self.data[[k for k, v in self.data.variables.items() if v.dims == ('observations',)]].to_dataframe()

    @abc.abstractmethod
    def generate_data(self, *args, **kwargs):
        """
        Should sample random data based on distribution and parameters.
        """
        pass

    @abc.abstractmethod
    def generate_params(self, *args, **kwargs):
        """
        Should generate all necessary parameters.
        """
        pass

    def load(self, path, group=""):
        """
        Loads pre-sampled data and parameters from specified HDF5 file
        :param path: the path to the HDF5 file
        :param group: the group inside the HDF5 file
        """
        path = os.path.expanduser(path)

        self.data = xr.open_dataset(
            path,
            group=os.path.join(group, "data"),
            engine=pkg_constants.XARRAY_NETCDF_ENGINE
        )
        self.params = xr.open_dataset(
            path,
            group=os.path.join(group, "params"),
            engine=pkg_constants.XARRAY_NETCDF_ENGINE
        )

        self.num_features = self.data.dims["features"]
        self.num_observations = self.data.dims["observations"]

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
            group=os.path.join(group, "data"),
            mode=mode,
            engine=pkg_constants.XARRAY_NETCDF_ENGINE
        )
        self.params.to_netcdf(
            path,
            group=os.path.join(group, "params"),
            mode="a",
            engine=pkg_constants.XARRAY_NETCDF_ENGINE
        )

    def data_to_anndata(self):
        """
        Converts the generated data into an anndata.AnnData object

        :return: anndata.AnnData object
        """
        adata = anndata.AnnData(self.data.X.values)
        for k, v in self.data.variables.items():
            if k == "X":
                continue
            if v.dims == ("observations",):
                adata.obs[k] = v.values
            elif v.dims[0] == "observations":
                adata.obsm[k] = v.values
            elif v.dims == ("features",):
                adata.var[k] = v.values
            elif v.dims[0] == "features":
                adata.varm[k] = v.values
            else:
                adata.uns[k] = v.values

        return adata

    def __copy__(self):
        retval = self.__class__()
        retval.num_observations = self.num_observations
        retval.num_features = self.num_features

        retval.data = self.data.copy()
        retval.params = self.params.copy()

        return retval

    def __str__(self):
        return "[%s.%s object at %s]:\ndata=%s\nparams=%s" % (
            type(self).__module__,
            type(self).__name__,
            hex(id(self)),
            str(self.data).replace("\n", "\n    "),
            str(self.params).replace("\n", "\n    "),
        )

    def __repr__(self):
        return self.__str__()
