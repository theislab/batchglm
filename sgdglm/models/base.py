import abc
from typing import Union, Any, Dict, Iterable, TypeVar, Type

import os

import xarray as xr

try:
    import anndata
except ImportError:
    anndata = None

import pkg_constants


class BasicInputData:
    """
    base class for all input data types
    """
    pass


class BasicModel(metaclass=abc.ABCMeta):

    @classmethod
    @abc.abstractmethod
    def param_shapes(cls) -> dict:
        """
        This method should return a dict of {parameter: (dim0_name, dim1_name, ..)} mappings
        for all parameters of this estimator.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def input_data(self) -> BasicInputData:
        """
        Get the input data of this model

        :return: the input data object
        """
        raise NotImplementedError()

    def to_xarray(self, parm: Union[str, list]):
        """
        Converts the specified parameters into an xr.Dataset or xr.DataArray object

        :param parm: string or list of strings specifying parameters which can be fetched by `self.get(params)`
        """
        # fetch data
        data = self.get(parm)

        # get shape of params
        shapes = self.param_shapes()

        if isinstance(parm, str):
            output = xr.DataArray(data, dims=shapes[parm])
        else:
            output = {key: (shapes[key], data[key]) for key in parm}
            output = xr.Dataset(output)

        return output

    def to_anndata(self, parm: list, adata: anndata.AnnData):
        """
        Converts the specified parameters into an anndata.AnnData object

        :param parm: string or list of strings specifying parameters which can be fetched by `self.get(params)`
        :param adata: the anndata.Anndata object to which the parameters will be appended
        """
        if isinstance(parm, str):
            parm = [parm]

        # fetch data
        data = self.get(parm)

        # get shape of params
        shapes = self.param_shapes()

        output = {key: (shapes[key], data[key]) for key in parm}
        for k, v in output.items():
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
            if k not in self.param_shapes():
                raise ValueError("Unknown parameter %s" % k)

        if isinstance(key, str):
            return self.__getattribute__(key)
        elif isinstance(key, Iterable):
            return {s: self.__getattribute__(s) for s in key}

    def __getitem__(self, item):
        return self.get(item)


_Estim_T = TypeVar('_Estim_T', bound='BasicEstimator')


class BasicEstimator(BasicModel, metaclass=abc.ABCMeta):

    def validate_data(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def initialize(self, **kwargs):
        """
        Initializes this estimator
        """
        pass

    @abc.abstractmethod
    def train(self, **kwargs):
        """
        Starts the training routine
        """
        pass

    @abc.abstractmethod
    def finalize(self, **kwargs) -> Type[_Estim_T]:
        """
        Clean up, free resources

        :return: some Estimator containing all necessary data
        """
        pass

    @property
    @abc.abstractmethod
    def loss(self, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def gradient(self, **kwargs):
        pass


class BasicSimulator(BasicModel, metaclass=abc.ABCMeta):
    num_observations: int
    num_features: int

    data: xr.Dataset
    params: xr.Dataset

    """
    Classes implementing `MatrixSimulator` should be able to generate a
    2D-matrix of sample data, as well as a dict of corresponding parameters.

    convention: N features with M observations each => (M, N) matrix
    """

    def __init__(self, num_observations=2000, num_features=10000):
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
