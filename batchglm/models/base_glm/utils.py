import logging
import math
from typing import Callable, List, Optional, Tuple, Union

import dask.array
import numpy as np
import pandas as pd
import patsy
import scipy.sparse

from .external import groupwise_solve_lm

logger = logging.getLogger("batchglm")


def generate_sample_description(
    num_observations: int,
    num_conditions: int,
    num_batches: int,
    intercept_scale: bool,
    shuffle_assignments: bool,
) -> Tuple[patsy.DesignMatrix, patsy.DesignMatrix, pd.DataFrame]:
    """Build a sample description.

    :param num_observations: Number of observations to simulate.
    :param num_conditions: number of conditions; will be repeated like [1,2,3,1,2,3]
    :param num_batches: number of conditions; will be repeated like [1,1,2,2,3,3]
    :param intercept_scale: If true, returns a single-coefficient design matrix (formula = "~1").
        If false, returns a design matrix identical to the loc model.
    :param shuffle_assignments: If true, shuffle the assignments in the xarray.
        UNSUPPORTED: Must be removed as it is disfunctional!!!
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
    sample_description = pd.DataFrame({"condition": conditions, "batch": batches})

    if shuffle_assignments:
        sample_description = sample_description.isel(
            observations=np.random.permutation(sample_description.observations.values)
        )

    sim_design_loc = patsy.dmatrix("~1+condition+batch", sample_description)

    if intercept_scale:
        sim_design_scale = patsy.dmatrix("~1", sample_description)
    else:
        sim_design_scale = sim_design_loc

    return sim_design_loc, sim_design_scale, sample_description

def closedform_glm_mean(
    x: Union[np.ndarray, scipy.sparse.csr_matrix, dask.array.core.Array],
    dmat: Union[np.ndarray, dask.array.core.Array],
    constraints: Optional[Union[np.ndarray, dask.array.core.Array]] = None,
    size_factors: Optional[np.ndarray] = None,
    link_fn: Optional[Callable] = None,
    inv_link_fn: Optional[Callable] = None,
):
    r"""
    Calculate a closed-form solution for the mean parameters of GLMs.

    :param x: The input data array
    :param dmat: some design matrix
    :param constraints: tensor (all parameters x dependent parameters)
        Tensor that encodes how complete parameter set which includes dependent
        parameters arises from indepedent parameters: all = <constraints, indep>.
        This form of constraints is used in vector generalized linear models (VGLMs).
    :param size_factors: size factors for X
    :param link_fn: linker function for GLM
    :param inv_link_fn: inverse linker function for GLM
    :return: tuple: (groupwise_means, mu, rmsd)
    """
    if size_factors is not None:
        x = np.divide(x, size_factors)

    def apply_fun(grouping):
        groupwise_means = np.asarray(
            np.vstack([np.mean(x[np.where(grouping == g)[0], :], axis=0) for g in np.unique(grouping)])
        )
        if link_fn is None:
            return groupwise_means
        else:
            return link_fn(groupwise_means)

    linker_groupwise_means, mu, rmsd, rank, s = groupwise_solve_lm(
        dmat=dmat, apply_fun=apply_fun, constraints=constraints
    )
    if inv_link_fn is not None:
        return inv_link_fn(linker_groupwise_means), mu, rmsd
    else:
        return linker_groupwise_means, mu, rmsd


def closedform_glm_scale(
    x: Union[np.ndarray, scipy.sparse.csr_matrix, dask.array.core.Array],
    design_scale: Union[np.ndarray, dask.array.core.Array],
    constraints: Optional[Union[np.ndarray, dask.array.core.Array]] = None,
    size_factors: Optional[np.ndarray] = None,
    groupwise_means: Optional[np.ndarray] = None,
    link_fn: Optional[Callable] = None,
    inv_link_fn: Optional[Callable] = None,
    compute_scales_fun: Optional[Callable] = None,
):
    r"""
    Calculate a closed-form solution for the scale parameters of GLMs.

    :param x: The sample data
    :param design_scale: design matrix for scale
    :param constraints: some design constraints
    :param size_factors: size factors for X
    :param groupwise_means: optional, in case if already computed this can be specified to spare double-calculation
    :param compute_scales_fun: TODO
    :param inv_link_fn: TODO
    :param link_fn: TODO
    :return: tuple (groupwise_scales, logphi, rmsd)
    """
    if size_factors is not None:
        x = x / size_factors

    # to circumvent nonlocal error
    provided_groupwise_means = groupwise_means

    def apply_fun(grouping):
        # Calculate group-wise means if not supplied. These are required for variance and MME computation.
        if provided_groupwise_means is None:
            gw_means = np.asarray(
                np.vstack([np.mean(x[np.where(grouping == g)[0], :], axis=0) for g in np.unique(grouping)])
            )
        else:
            gw_means = provided_groupwise_means

        # calculated variance via E(x)^2 or directly depending on whether `mu` was specified
        if isinstance(x, scipy.sparse.csr_matrix):
            expect_xsq = np.asarray(
                np.vstack(
                    [
                        np.asarray(np.mean(x[np.where(grouping == g)[0], :].power(2), axis=0))
                        for g in np.unique(grouping)
                    ]
                )
            )
        else:
            expect_xsq = np.vstack(
                [np.mean(np.square(x[np.where(grouping == g)[0], :]), axis=0) for g in np.unique(grouping)]
            )
        expect_x_sq = np.square(gw_means)
        variance = expect_xsq - expect_x_sq

        if compute_scales_fun is not None:
            groupwise_scales = compute_scales_fun(variance, gw_means)
        else:
            groupwise_scales = variance

        if link_fn is not None:
            return link_fn(groupwise_scales)
        else:
            return groupwise_scales

    linker_groupwise_scales, scaleparam, rmsd, rank, _ = groupwise_solve_lm(
        dmat=design_scale, apply_fun=apply_fun, constraints=constraints
    )
    if inv_link_fn is not None:
        return inv_link_fn(linker_groupwise_scales), scaleparam, rmsd
    else:
        return linker_groupwise_scales, scaleparam, rmsd
