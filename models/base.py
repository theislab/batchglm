import abc
from typing import Union, Any, Dict, Iterable

import os

import xarray as xr

import pkg_constants


class BasicModel(metaclass=abc.ABCMeta):

    @classmethod
    @abc.abstractmethod
    def params(cls) -> dict:
        """
        This method should return a dict of {parameter: (dim0_name, dim1_name, ..)} mappings
        for all parameters of this estimator.
        """
        raise NotImplementedError()

    def to_xarray(self, params: list):
        # fetch data
        data = self.get(params)

        # get shape of params
        shapes = self.params()

        output = {key: (shapes[key], data[key]) for key in params}
        output = xr.Dataset(output)

        return output

    @abc.abstractmethod
    def export_params(self, append_to=None, **kwargs):
        """
        Exports this model in another format

        :param append_to: If specified, the parameters will be appended to this data set
        :return: data set containing all necessary parameters of this model.

            If `append_to` is specified, the return value will be of type `type(append_to)`.

            Otherwise, a xarray.Dataset will be returned.
        """
        pass

    def get(self, key: Union[str, Iterable]) -> Union[Any, Dict[str, Any]]:
        """
        Returns the values specified by key.

        :param key: Either a string or an iterable list/set/tuple/etc. of strings
        :return: Single array if `key` is a string or a dict {k: value} of arrays if `key` is a collection of strings
        """
        for k in list(key):
            if k not in self.params():
                raise ValueError("Unknown parameter %s" % k)

        if isinstance(key, str):
            return self.__getattribute__(key)
        elif isinstance(key, Iterable):
            return {s: self.__getattribute__(s) for s in key}

    def __getitem__(self, item):
        return self.get(item)


class BasicEstimator(BasicModel, metaclass=abc.ABCMeta):
    input_data: any
    loss: any

    def __init__(self, input_data):
        self.input_data = input_data

    def validate_data(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def initialize(self, **kwargs):
        pass

    @abc.abstractmethod
    def train(self, **kwargs):
        pass


class BasicSimulator(BasicModel, metaclass=abc.ABCMeta):
    num_samples: int
    num_genes: int

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
            engine=pkg_constants.XARRAY_NETCDF_ENGINE
        )
        self.params.to_netcdf(
            path,
            group=os.path.join(group, "params"),
            mode="a",
            engine=pkg_constants.XARRAY_NETCDF_ENGINE
        )

    def __copy__(self):
        retval = self.__class__()
        retval.num_samples = self.num_samples
        retval.num_genes = self.num_genes

        retval.data = self.data.copy()
        retval.params = self.params.copy()

        return retval
