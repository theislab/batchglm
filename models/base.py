import abc
import os

import xarray as xr

# _XARRAY_NETCDF_ENGINE = "netcdf4"
_XARRAY_NETCDF_ENGINE = "h5netcdf"


class BasicEstimator(metaclass=abc.ABCMeta):
    input_data: xr.Dataset
    loss: any

    def __init__(self, input_data: xr.Dataset):
        self.input_data = input_data

    @abc.abstractmethod
    def validate_data(self, **kwargs):
        pass

    @abc.abstractmethod
    def initialize(self, **kwargs):
        pass

    @abc.abstractmethod
    def train(self, **kwargs):
        pass


class BasicSimulator(metaclass=abc.ABCMeta):
    data: xr.Dataset
    params: xr.Dataset

    """
    Classes implementing `MatrixSimulator` should be able to generate a
    2D-matrix of sample data, as well as a dict of corresponding parameters.

    convention: N genes with M samples each => (M, N) matrix
    """

    def __init__(self, num_samples=2000, num_genes=10000):
        self.num_samples = num_samples
        self.num_genes = num_genes

        self.data = xr.Dataset()
        self.params = xr.Dataset()

    def generate(self):
        """
        First generates the parameter set, then samples random data using these parameters
        """
        self.generate_params()
        self.generate_data()

    @abc.abstractmethod
    def generate_data(self, *args, **kwargs):
        """
        Should sample random data using the pre-defined / sampled parameters
        """
        pass

    @abc.abstractmethod
    def generate_params(self, *args, **kwargs):
        """
        Should generate all necessary parameters
        """
        pass

    def load(self, path, group=""):
        """
        Loads pre-sampled data and parameters from specified HDF5 file
        :param path: the path to the HDF5 file
        :param group: the group inside the HDF5 file
        """
        path = os.path.expanduser(path)

        self.data = xr.open_dataset(path, group=os.path.join(group, "data"), engine="h5netcdf")
        self.params = xr.open_dataset(path, group=os.path.join(group, "params"), engine="h5netcdf")

        self.num_genes = self.data.dims["genes"]
        self.num_samples = self.data.dims["samples"]

    def save(self, path, group="", append=False):
        """
        Saves parameters and sampled data to specified file in HDF5 format
        :param path: the path to the target file where the data will be saved
        :param group: the group+ inside the HDF5 file where the data will be saved
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
            engine=_XARRAY_NETCDF_ENGINE
        )
        self.params.to_netcdf(
            path,
            group=os.path.join(group, "params"),
            mode="a",
            engine=_XARRAY_NETCDF_ENGINE
        )
