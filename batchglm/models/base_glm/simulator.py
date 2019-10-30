import abc
import math
import numpy as np
import pandas
import patsy
from typing import Union, Tuple

from .model import _ModelGLM
from .external import _SimulatorBase


def generate_sample_description(
        num_observations,
        num_conditions: int = 2,
        num_batches: int = 4,
        shuffle_assignments=False
) -> Tuple[patsy.DesignMatrix, pandas.DataFrame]:
    """ Build a sample description.

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
    sample_description = pandas.DataFrame({
        "condition": conditions,
        "batch": batches
    })

    if shuffle_assignments:
        sample_description = sample_description.isel(
            observations=np.random.permutation(sample_description.observations.values)
        )

    return patsy.dmatrix("~1+condition+batch", sample_description), sample_description


class _SimulatorGLM(_SimulatorBase, metaclass=abc.ABCMeta):
    """
    Simulator for Generalized Linear Models (GLMs).
    """
    design_loc: patsy.design_info.DesignMatrix
    design_scale: patsy.design_info.DesignMatrix
    sample_description: pandas.DataFrame

    def __init__(
            self,
            model: Union[_ModelGLM, None],
            num_observations,
            num_features
    ):
        _SimulatorBase.__init__(
            self=self,
            model=model,
            num_observations=num_observations,
            num_features=num_features
        )
        self.sim_design_loc = None
        self.sim_design_scale = None
        self.sample_description = None
        self.sim_a_var = None
        self.sim_b_var = None
        self._size_factors = None

    def generate_sample_description(
            self,
            num_conditions=2,
            num_batches=4,
            intercept_scale: bool = False,
            **kwargs
    ):
        self.sim_design_loc, self.sample_description = generate_sample_description(
            self.nobs,
            num_conditions=num_conditions,
            num_batches=num_batches,
            **kwargs
        )
        if intercept_scale:
            self.sim_design_scale = patsy.dmatrix("~1", self.sample_description)
        else:
            self.sim_design_scale = self.sim_design_loc

    def _generate_params(
            self,
            *args,
            rand_fn_ave=None,
            rand_fn=None,
            rand_fn_loc=None,
            rand_fn_scale=None,
            **kwargs
    ):
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

        self.sim_a_var = np.concatenate([
            self.link_loc(np.expand_dims(rand_fn_ave([self.nfeatures]), axis=0)),  # intercept
            rand_fn_loc((self.sim_design_loc.shape[1] - 1, self.nfeatures))
        ], axis=0)
        self.sim_b_var = np.concatenate([
            rand_fn_scale((self.sim_design_scale.shape[1], self.nfeatures))
        ], axis=0)

    @property
    def size_factors(self):
        return self._size_factors

    @property
    def a_var(self):
        return self.sim_a_var

    @property
    def b_var(self):
        return self.sim_b_var

    @property
    def design_loc(self) -> np.ndarray:
        return self.sim_design_loc

    @property
    def design_scale(self) -> np.ndarray:
        return self.sim_design_scale

    @property
    def constraints_loc(self):
        return np.identity(n=self.a_var.shape[0])

    @property
    def constraints_scale(self):
        return np.identity(n=self.b_var.shape[0])
