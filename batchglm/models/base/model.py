import abc
from typing import Union, Any, Dict, Iterable
import logging

import xarray as xr

try:
    import anndata
except ImportError:
    anndata = None

from .input import _InputData_Base

logger = logging.getLogger(__name__)


class _Model_Base(metaclass=abc.ABCMeta):
    r"""
    Model base class
    """

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
    def input_data(self) -> _InputData_Base:
        """
        Get the input data of this model

        :return: the input data object
        """
        raise NotImplementedError()

    def to_xarray(self, parm: Union[str, list], coords=None):
        """
        Converts the specified parameters into an xr.Dataset or xr.DataArray object

        :param parm: string or list of strings specifying parameters which can be fetched by `self.get(params)`
        :param coords: optional dict-like object with arrays corresponding to dimension names
        """
        # fetch data
        data = self.get(parm)

        # get shape of params
        shapes = self.param_shapes()

        if isinstance(parm, str):
            output = xr.DataArray(data, dims=shapes[parm])
            if coords is not None:
                for i in output.dims:
                    if i in coords:
                        output.coords[i] = coords[i]
        else:
            output = {key: (shapes[key], data[key]) for key in parm}
            output = xr.Dataset(output)
            if coords is not None:
                for i in output.dims:
                    if i in coords.coords:
                        output.coords[i] = coords[i]

            #TODO: output['a_var'] does not have the correct dimension names!

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


class _Model_XArray_Base():
    _input_data: _InputData_Base
    params: xr.Dataset

    def __init__(self, input_data: _InputData_Base, params: xr.Dataset):
        self._input_data = input_data
        self.params = params

    @property
    def input_data(self) -> _InputData_Base:
        return self._input_data

    def __str__(self):
        return "[%s.%s object at %s]: data=%s" % (
            type(self).__module__,
            type(self).__name__,
            hex(id(self)),
            self.params
        )

    def __repr__(self):
        return self.__str__()

