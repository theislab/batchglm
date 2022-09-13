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


def parse_design(
    design_matrix: Union[pd.DataFrame, patsy.design_info.DesignMatrix, dask.array.core.Array, np.ndarray],
    param_names: List[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    r"""
    Parser for design matrices.

    :param design_matrix: Design matrix.
    :param param_names:
        Optional coefficient names for design_matrix.
        Ignored if design_matrix is pd.DataFrame or patsy.design_info.DesignMatrix.
    :return: Tuple[np.ndarray, List[str]] containing the design matrix and the parameter names.
    :raise AssertionError: if the type of design_matrix is not understood.
    :raise AssertionError: if length of provided param_names is not equal to number of coefficients in design_matrix.
    :raise ValueError: if param_names is None when type of design_matrix is numpy.ndarray or dask.array.core.Array.
    """
    if isinstance(design_matrix, (pd.DataFrame, patsy.design_info.DesignMatrix)) and param_names is not None:
        logger.warning(f"The provided param_names are ignored as the design matrix is of type {type(design_matrix)}.")

    if isinstance(design_matrix, patsy.design_info.DesignMatrix):
        dmat = np.asarray(design_matrix)
        params = design_matrix.design_info.column_names
    elif isinstance(design_matrix, pd.DataFrame):
        dmat = np.asarray(design_matrix)
        params = design_matrix.columns.tolist()
    elif isinstance(design_matrix, dask.array.core.Array):
        dmat = design_matrix.compute()
        params = param_names
    elif isinstance(design_matrix, np.ndarray):
        dmat = design_matrix
        params = param_names
    else:
        raise AssertionError(f"Datatype for design_matrix not understood: {type(design_matrix)}")
    if params is None:
        raise ValueError("Provide names when passing design_matrix as np.ndarray or dask.array.core.Array!")
    assert len(params) == dmat.shape[1], (
        "Length of provided param_names is not equal to " "number of coefficients in design_matrix."
    )
    return dmat, params


def parse_constraints(
    dmat: np.ndarray,
    dmat_par_names: List[str],
    constraints: Optional[Union[np.ndarray, dask.array.core.Array]] = None,
    constraint_par_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    r"""
    Parser for constraint matrices.

    :param dmat: Design matrix.
    :param constraints: Constraint matrix.
    :param constraint_par_names: Optional coefficient names for constraints.
    :return: Tuple[np.ndarray, List[str]] containing the constraint matrix and the parameter names.
    :raise AssertionError: if the type of given design / contraint matrix is not np.ndarray or dask.array.core.Array.
    """
    assert isinstance(dmat, np.ndarray), "dmat must be provided as np.ndarray."
    if constraints is None:
        constraints = np.identity(n=dmat.shape[1])
        constraint_params = dmat_par_names
    else:
        if isinstance(constraints, dask.array.core.Array):
            constraints = constraints.compute()
        assert isinstance(constraints, np.ndarray), "contraints must be np.ndarray or dask.array.core.Array."
        # Cannot use all parameter names if constraint matrix is not identity: Make up new ones.
        # Use variable names that can be mapped (unconstrained).
        if constraint_par_names is not None:
            assert len(constraint_params) == len(constraint_par_names)
            constraint_params = constraint_par_names
        else:
            constraint_params = [
                "var_" + str(i)
                if np.sum(constraints[:, i] != 0) > 1
                else dmat_par_names[np.where(constraints[:, i] != 0)[0][0]]
                for i in range(constraints.shape[1])
            ]
        assert constraints.shape[0] == dmat.shape[1], "constraint dimension mismatch"

    return constraints, constraint_params


def closedform_glm_mean(
    x: Union[np.ndarray, scipy.sparse.csr_matrix, dask.array.core.Array],
    dmat: Union[np.ndarray, dask.array.core.Array],
    constraints: Union[np.ndarray, dask.array.core.Array],
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
    constraints: Union[np.ndarray, dask.array.core.Array],
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
