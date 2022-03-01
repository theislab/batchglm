import abc
import logging
from typing import Any, Callable, Dict, Iterable, Optional, Union

import dask.array
import numpy as np

try:
    import anndata
except ImportError:
    anndata = None

import scipy

from .external import pkg_constants
from .input import InputDataGLM
from .utils import generate_sample_description, parse_constraints, parse_design

logger = logging.getLogger(__name__)


class _ModelGLM(metaclass=abc.ABCMeta):
    """
    Generalized Linear Model base class.

    Every GLM contains parameters for a location and a scale model
    in a parameter specific linker space and a design matrix for
    each location and scale model.
    input_data : batchglm.models.base_glm.input.InputData
        Input data
    """

    _design_loc: np.ndarray = None
    _design_scale: np.ndarray = None
    _constraints_loc: np.ndarray = None
    _constraints_scale: np.ndarray = None
    _design_loc_names: np.ndarray = None
    _design_scale_names: np.ndarray = None
    _loc_names: np.ndarray = None
    _scale_names: np.ndarray = None
    _x: np.ndarray = None
    _size_factors: np.ndarray = None
    _theta_location: np.ndarray = None
    _theta_scale: np.ndarray = None
    _theta_location_getter: Callable = lambda x: x._theta_location
    _theta_scale_getter: Callable = lambda x: x._theta_scale
    _cast_dtype: str = None
    _loc_names: np.ndarray = None
    _scale_names: np.ndarray = None
    _chunk_size_cells: int = None
    _chunk_size_genes: int = None

    def __init__(
        self,
        input_data: Optional[InputDataGLM] = None,
    ):
        """
        Create a new _ModelGLM object.

        :param input_data: Input data for the model

        """
        if input_data is not None:
            self.extract_input_data(input_data)
    
    def extract_input_data(self, input_data: InputDataGLM):
        self._design_loc = input_data.design_loc
        self._design_scale = input_data.design_scale
        self._constraints_loc = input_data.constraints_loc
        self._constraints_scale = input_data.constraints_scale
        self._design_loc_names = input_data.design_loc_names
        self._design_scale_names = input_data.design_scale_names
        self._loc_names = input_data.loc_names
        self._scale_names = input_data.scale_names
        self._x = input_data.x
        self._size_factors = input_data.size_factors
        self._cast_dtype = input_data.cast_dtype
        self._chunk_size_genes = input_data.chunk_size_genes
        self._chunk_size_cells = input_data.chunk_size_cells

    @property
    def chunk_size_cells(self) -> int:
        return self._chunk_size_cells

    @property
    def chunk_size_genes(self) -> int:
        return self._chunk_size_genes

    @property
    def cast_dtype(self) -> str:
        return self._cast_dtype

    @property
    def design_loc(self) -> Union[np.ndarray, dask.array.core.Array]:
        """location design matrix"""
        return self._design_loc

    @property
    def design_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        """scale design matrix"""
        return self._design_scale

    @property
    def constraints_loc(self) -> Union[np.ndarray, dask.array.core.Array]:
        """constrainted location design matrix"""
        return self._constraints_loc

    @property
    def constraints_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        """constrained scale design matrix"""
        return self._constraints_scale

    @property
    def design_loc_names(self) -> list:
        """column names from location design matrix"""
        return self._design_loc_names

    @property
    def design_scale_names(self) -> list:
        """column names from scale design matrix"""
        return self._design_scale_names

    @property
    def loc_names(self) -> list:
        """column names from constratined location design matrix"""
        return self._loc_names

    @property
    def scale_names(self) -> list:
        """column names from constrained scale design matrix"""
        return self._scale_names

    @abc.abstractmethod
    def eta_loc(self) -> Union[np.ndarray, dask.array.core.Array]:
        pass

    @property
    def eta_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        """eta from scale model"""
        eta = np.matmul(self.design_scale, self.theta_scale_constrained)
        eta = self.np_clip_param(eta, "eta_scale")
        return eta

    @property
    def location(self):
        """the inverse link function applied to eta for the location model (i.e the fitted location)"""
        return self.inverse_link_loc(self.eta_loc)

    @property
    def scale(self):
        """the inverse link function applied to eta for the scale model (i.e the fitted location)"""
        return self.inverse_link_scale(self.eta_scale)

    @abc.abstractmethod
    def eta_loc_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        """
        Method to be implemented that allows fast access to a given observation's eta in the location model
        :param j: The index of the observation sought
        """
        pass

    def eta_scale_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        """
        Allows fast access to a given observation's eta in the location model
        :param j: The index of the observation sought
        """
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        return np.matmul(self.design_scale, self.theta_scale_constrained[:, j])

    def location_j(self, j):
        """
        Allows fast access to a given observation's fitted location
        :param j: The index of the observation sought
        """
        return self.inverse_link_loc(self.eta_loc_j(j=j))

    def scale_j(self, j):
        """
        Allows fast access to a given observation's fitted scale
        :param j: The index of the observation sought
        """
        return self.inverse_link_scale(self.eta_scale_j(j=j))

    @property
    def xh_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        return np.matmul(self.design_scale, self.constraints_scale)

    @property
    def x(self):
        """Get the counts data matrix."""
        return self._x

    def x_j(self, j):
        """
        Allows fast access to a given observation's x data
        :param j: The index of the observation sought
        """
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        return self.x[:, j]

    @property
    def num_observations(self):
        """Number of observations derived from x."""
        return self.x.shape[0]

    @property
    def num_features(self):
        """Number of features derived from x."""
        return self.x.shape[1]

    @property
    def num_loc_params(self):
        """Number of columns of the location design matrix subject to constraints"""
        return self.constraints_loc.shape[1]

    @property
    def num_scale_params(self):
        """Number of columns of the scale design matrix subject to constraints"""
        return self.constraints_scale.shape[1]

    @property
    def size_factors(self) -> Union[np.ndarray, None]:
        """Constant scale factors of the mean model in the linker space"""
        return self._size_factors

    @property
    def theta_location(self) -> np.ndarray:
        """Fitted location model parameters"""
        return self._theta_location_getter()

    @property
    def theta_scale(self) -> np.ndarray:
        """Fitted scale model parameters"""
        return self._theta_scale_getter()

    @property
    def theta_location_constrained(self) -> Union[np.ndarray, dask.array.core.Array]:
        """dot product of location constraints with location parameter giving new constrained parameters"""
        return np.dot(self.constraints_loc, self.theta_location)

    @property
    def theta_scale_constrained(self) -> Union[np.ndarray, dask.array.core.Array]:
        """dot product of scale constraints with scale parameter giving new constrained parameters"""
        return np.dot(self.constraints_scale, self.theta_scale)

    @abc.abstractmethod
    def link_loc(self, data):
        """link function for location model"""
        pass

    @abc.abstractmethod
    def link_scale(self, data):
        """link function for scale model"""
        pass

    @abc.abstractmethod
    def inverse_link_loc(self, data):
        """inverse link function for location model"""
        pass

    @abc.abstractmethod
    def inverse_link_scale(self, data):
        """inverse link function for scale model"""
        pass

    def get(self, key: Union[str, Iterable]) -> Union[Any, Dict[str, Any]]:
        """
        Returns the values specified by key.

        :param key: Either a string or an iterable list/set/tuple/etc. of strings
        :return: Single array if `key` is a string or a dict {k: value} of arrays if `key` is a collection of strings
        """
        if isinstance(key, str):
            attrib = self.__getattribute__(key)
        elif isinstance(key, Iterable):
            attrib = {s: self.__getattribute__(s) for s in key}
        return attrib

    # parameter contraints:

    def np_clip_param(self, param, name):
        bounds_min, bounds_max = self.param_bounds(param.dtype)
        return np.clip(param, bounds_min[name], bounds_max[name])

    def param_bounds(self, dtype):

        dtype = np.dtype(dtype)
        # dmin = np.finfo(dtype).min
        dmax = np.finfo(dtype).max
        dtype = dtype.type
        sf = dtype(pkg_constants.ACCURACY_MARGIN_RELATIVE_TO_LIMIT)

        return self.bounds(sf, dmax, dtype)

    @abc.abstractmethod
    def bounds(self, sf, dmax, dtype) -> Dict[str, Any]:
        pass

    # simulator:

    @abc.abstractmethod
    def rand_fn_ave(self) -> Optional[Callable]:
        pass

    @abc.abstractmethod
    def rand_fn(self) -> Optional[Callable]:
        pass

    @abc.abstractmethod
    def rand_fn_loc(self) -> Optional[Callable]:
        pass

    @abc.abstractmethod
    def rand_fn_scale(self) -> Optional[Callable]:
        pass

    def generate_params(
        self,
        n_vars: int,
        rand_fn_ave=None,
        rand_fn=None,
        rand_fn_loc=None,
        rand_fn_scale=None,
        **kwargs
    ):
        """
        Generate all necessary parameters. TODO: make this documentation better!!!

        :param rand_fn_ave: function which generates random numbers for intercept.
            Takes one location parameter of intercept distribution across features.
        :param rand_fn: random function taking one argument `shape`.
        :param rand_fn_loc: random function taking one argument `shape`.
            If not provided, will use `rand_fn` instead.
            This function generates location model parameters in inverse linker space,
            ie. these parameter will be log transformed if a log linker function is used!
            Values below 1e-08 will be set to 1e-08 to map them into the positive support.
        :param rand_fn_scale: random function taking one argument `shape`.
            If not provided, will use `rand_fn` instead.
            This function generates scale model parameters in inverse linker space,
            ie. these parameter will be log transformed if a log linker function is used!
            Values below 1e-08 will be set to 1e-08 to map them into the positive support.
        """

        if rand_fn_ave is None:
            rand_fn_ave = self.rand_fn_ave
            if rand_fn_ave is None:
                raise ValueError("rand_fn_ave must not be None!")
        if rand_fn is None:
            rand_fn = self.rand_fn
        if rand_fn_loc is None:
            rand_fn_loc = self.rand_fn_loc
        if rand_fn_scale is None:
            rand_fn_scale = self.rand_fn_scale
        if rand_fn is None and rand_fn_loc is None:
            raise ValueError("rand_fn and rand_fn_loc must not be both None!")
        if rand_fn is None and rand_fn_scale is None:
            raise ValueError("rand_fn and rand_fn_scale must not be both None!")

        if rand_fn_loc is None:
            rand_fn_loc = rand_fn
        if rand_fn_scale is None:
            rand_fn_scale = rand_fn

        _design_loc, _design_scale, _ = generate_sample_description(**kwargs)

        self._theta_location = np.concatenate(
            [
                self.link_loc(np.expand_dims(rand_fn_ave([n_vars]), axis=0)),  # intercept
                rand_fn_loc((_design_loc.shape[1] - 1, n_vars)),
            ],
            axis=0,
        )
        self._theta_scale = np.concatenate([rand_fn_scale((_design_scale.shape[1], n_vars))], axis=0)

        return _design_loc, _design_scale


    def generate(
        self,
        n_obs: int,
        n_vars: int,
        num_conditions: int = 2,
        num_batches: int = 4,
        intercept_scale: bool = False,
        shuffle_assignments: bool = False,
        sparse: bool = False,
        as_dask: bool = True,
        **kwargs,
    ):
        """
        First generates the parameter set, then observations random data using these parameters.

        :param sparse: Description of parameter `sparse`.
        """
        _design_loc, _design_scale = self.generate_params(
            n_vars=n_vars,
            num_observations=n_obs,
            num_conditions=num_conditions,
            num_batches=num_batches,
            intercept_scale=intercept_scale,
            shuffle_assignments=shuffle_assignments,
            **kwargs
        )

        # we need to do this explicitly here in order to generate data
        self._constraints_loc = np.identity(n=_design_loc.shape[1])
        self._constraints_scale = np.identity(n=_design_scale.shape[1])
        self._design_loc = _design_loc
        self._design_scale = _design_scale

        # generate data
        data_matrix = self.generate_data().astype(self.cast_dtype)
        if sparse:
            data_matrix = scipy.sparse.csr_matrix(data_matrix)

        input_data = InputDataGLM(
            data=data_matrix,
            design_loc=_design_loc,
            design_scale=_design_scale,
            as_dask=as_dask
        )
        self.extract_input_data(input_data)

    @abc.abstractmethod
    def generate_data(self):
        """
        Should sample random data based on distribution and parameters.

        :param type args: TODO.
        :param type kwargs: TODO.
        """
        pass

    def __getitem__(self, item):
        return self.get(item)
