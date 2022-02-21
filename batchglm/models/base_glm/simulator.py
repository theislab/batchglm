import abc
import math
from typing import Tuple, Union

import numpy as np
import pandas
import patsy
import scipy

from .external import _SimulatorBase
from .input import InputDataGLM
from .model import _ModelGLM
import logging

logger = logging.getLogger(__name__)


def generate_sample_description(
    num_observations, num_conditions: int = 2, num_batches: int = 4, shuffle_assignments=False
) -> Tuple[patsy.DesignMatrix, pandas.DataFrame]:
    """Build a sample description.

    :param num_observations: Number of observations to simulate.
    :param num_conditions: number of conditions; will be repeated like [1,2,3,1,2,3]
    :param num_batches: number of conditions; will be repeated like [1,1,2,2,3,3]
    """
    if num_conditions == 0:
        num_conditions = 1
    if num_batches == 0:
        num_batches = 1

    # condition column
    reps_conditions = math.ceil(num_observations / num_conditions)
    conditions = np.squeeze(np.tile([np.arange(num_conditions)], reps_conditions))
    conditions = conditions[range(num_observations)].astype(str)

    # batch column
    reps_batches = math.ceil(num_observations / num_batches)
    batches = np.repeat(range(num_batches), reps_batches)
    batches = batches[range(num_observations)].astype(str)
    sample_description = pandas.DataFrame({"condition": conditions, "batch": batches})

    if shuffle_assignments:
        sample_description = sample_description.isel(
            observations=np.random.permutation(sample_description.observations.values)
        )

    return patsy.dmatrix("~1+condition+batch", sample_description), sample_description


class _SimulatorGLM(_SimulatorBase, metaclass=abc.ABCMeta):
    """
    Simulator for Generalized Linear Models (GLMs).
    Simulator base class.

    Classes implementing `BasicSimulator` should be able to generate a
    2D-matrix of sample data, as well as a dict of corresponding parameters.

    convention: N features with M observations each => (M, N) matrix

    Attributes
    ----------
    sim_design_loc : patsy.DesignMatrix
        Simulated design martix for location model
    sim_design_scale : patsy.DesignMatrix
        Simulated design martix for scale model
    sample_description : pandas.DataFrame
        A dataframe of data in terms of batch and condition
    sim_theta_location : np.ndarray
        Location model parameters
    sim_theta_scale : np.ndarray
        Scale model parameters
    """

    sim_design_loc: patsy.DesignMatrix = None
    sim_design_scale: patsy.DesignMatrix = None
    sample_description: pandas.DataFrame = None
    sim_theta_location: np.ndarray = None
    sim_theta_scale: np.ndarray = None

    def __init__(self, num_observations, num_features):
        """
        Create a new _SimulatorGLM object.

        :param num_observations: Number of observations
        :param num_features: Number of featurews
        """
        _SimulatorBase.__init__(self=self, num_observations=num_observations, num_features=num_features)

    def generate_sample_description(self, num_conditions=2, num_batches=4, intercept_scale: bool = False, **kwargs):
        """
        Generate a sample description for the simulator including patsy design matrices and fake batch/condition data

        :param num_conditions: Number of conditions for the design matrix.
        :param num_batches: Number of batches for the design matrix.
        :param intercept_scale: Whether or not to provide an intercept for the scale model (otherwise just use the location design matrix).
        """
        self.sim_design_loc, self.sample_description = generate_sample_description(
            self.nobs, num_conditions=num_conditions, num_batches=num_batches, **kwargs
        )
        if intercept_scale:
            self.sim_design_scale = patsy.dmatrix("~1", self.sample_description)
        else:
            self.sim_design_scale = self.sim_design_loc

    def _generate_params(self, *args, rand_fn_ave=None, rand_fn=None, rand_fn_loc=None, rand_fn_scale=None, **kwargs):
        """
        Generate all necessary parameters

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
            raise ValueError("rand_fn_ave must not be None!")
        if rand_fn is None and rand_fn_loc is None:
            raise ValueError("rand_fn and rand_fn_loc must not be both None!")
        if rand_fn is None and rand_fn_scale is None:
            raise ValueError("rand_fn and rand_fn_scale must not be both None!")

        if rand_fn_loc is None:
            rand_fn_loc = rand_fn
        if rand_fn_scale is None:
            rand_fn_scale = rand_fn

        if self.sim_design_loc is None:
            self.generate_sample_description()
        if self.sim_design_scale is None:
            self.sim_design_scale = self.sim_design_loc

        self.sim_theta_location = np.concatenate(
            [
                self.link_loc(np.expand_dims(rand_fn_ave([self.nfeatures]), axis=0)),  # intercept
                rand_fn_loc((self.sim_design_loc.shape[1] - 1, self.nfeatures)),
            ],
            axis=0,
        )
        self.sim_theta_scale = np.concatenate([rand_fn_scale((self.sim_design_scale.shape[1], self.nfeatures))], axis=0)

    def assemble_input_data(self, data_matrix: np.ndarray, sparse: bool):
        if sparse:
            data_matrix = scipy.sparse.csr_matrix(data_matrix)
        self.input_data = InputDataGLM(
            data=data_matrix,
            design_loc=self.sim_design_loc,
            design_scale=self.sim_design_scale,
            design_loc_names=None,
            design_scale_names=None,
        )

    @property
    def theta_location(self):
        """simulated location model parameters"""
        return self.sim_theta_location

    @property
    def theta_scale(self):
        """simulated scale model parameters"""
        return self.sim_theta_scale

    @property
    def design_loc(self) -> Union[patsy.design_info.DesignMatrix, np.ndarray]:
        """simulated location model design matrix"""
        return self.sim_design_loc

    @property
    def design_scale(self) -> Union[patsy.design_info.DesignMatrix, np.ndarray]:
        """simulated scale model design matrix"""
        return self.sim_design_scale

    @property
    def constraints_loc(self):
        """simulated constraints on location model"""
        return np.identity(n=self.theta_location.shape[0])

    @property
    def constraints_scale(self):
        """simulated constraints on scale model"""
        return np.identity(n=self.theta_scale.shape[0])

    def param_bounds(self, dtype):
        """method to be implemented that allows models to constrain certain parameters like means or fitted coefficients"""
        pass

    def eta_loc_j(self, j) -> np.ndarray:
        """method to be implemented that allows fast access to a given observation's eta"""
        pass

    def np_clip_param(self, param, name):
        # TODO: inherit this from somewhere?
        bounds_min, bounds_max = self.param_bounds(param.dtype)
        return np.clip(param, bounds_min[name], bounds_max[name])
