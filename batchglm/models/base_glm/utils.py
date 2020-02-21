from typing import List, Tuple, Union

try:
    import anndata
except ImportError:
    anndata = None

import dask.array
import numpy as np
import pandas as pd
import patsy
import scipy.sparse

from .external import groupwise_solve_lm


def parse_design(
        design_matrix: Union[pd.DataFrame, patsy.design_info.DesignMatrix],
        param_names: List[str] = None
) -> Tuple:
    r"""
    Parser for design matrices.

    :param design_matrix: Design matrix.
    :param param_names:
    :return: Tuple containing the design matrix and the parameter names.
    """
    if isinstance(design_matrix, patsy.design_info.DesignMatrix):
        dmat = np.asarray(design_matrix)
        params = design_matrix.design_info.column_names
    elif isinstance(design_matrix, pd.DataFrame):
        dmat = np.asarray(design_matrix)
        params = design_matrix.columns
    elif isinstance(design_matrix, np.ndarray):
        dmat = design_matrix
        params = None
    elif isinstance(design_matrix, dask.array.core.Array):
        dmat = design_matrix.compute()
        params = None
    else:
        raise ValueError("type %s not recognized" % type(design_matrix))

    if param_names is not None:
        if params is None:
            assert len(param_names) == dmat.shape[1]
            params = param_names

    return dmat, params


def parse_constraints(
        dmat: np.ndarray,
        dmat_par_names: List[str],
        constraints: Union[np.ndarray, dask.array.core.Array] = None,
        constraint_par_names: list = None
) -> Tuple:
    r"""
    Parser for constraint matrices.

    :param dmat: Design matrix.
    :param constraints: Constraint matrix.
    :param constraint_par_names: list of coords for xr.DataArray
    :return: Tuple containing the constraint matrix and the parameter names.
    """
    if constraints is None:
        constraints = np.identity(n=dmat.shape[1])
        constraint_params = dmat_par_names
    else:
        if isinstance(constraints, dask.array.core.Array):
            constraints = constraints.compute()
        # Cannot use all parameter names if constraint matrix is not identity: Make up new ones.
        # Use variable names that can be mapped (unconstrained).
        constraint_params = ["var_"+str(i) if np.sum(constraints[:, i] != 0) > 1
                             else dmat_par_names[np.where(constraints[:, i] != 0)[0][0]]
                             for i in range(constraints.shape[1])]
        assert constraints.shape[0] == dmat.shape[1], "constraint dimension mismatch"

    if constraint_par_names is not None:
        assert len(constraint_params) == len(constraint_par_names)
        constraint_params = constraint_par_names

    return constraints, constraint_params


def closedform_glm_mean(
        x: Union[np.ndarray, scipy.sparse.csr_matrix],
        dmat: np.ndarray,
        constraints=None,
        size_factors=None,
        link_fn: Union[callable, None] = None,
        inv_link_fn: Union[callable, None] = None
):
    r"""
    Calculates a closed-form solution for the mean parameters of GLMs.
    :param x: The input data array
    :param dmat: some design matrix
    :param constraints: tensor (all parameters x dependent parameters)
        Tensor that encodes how complete parameter set which includes dependent
        parameters arises from indepedent parameters: all = <constraints, indep>.
        This form of constraints is used in vector generalized linear models (VGLMs).
    :param size_factors: size factors for X
    :param link_fn: linker function for GLM
    :return: tuple: (groupwise_means, mu, rmsd)
    """
    if size_factors is not None:
        x = np.divide(x, size_factors)

    def apply_fun(grouping):
        groupwise_means = np.asarray(np.vstack([
            np.mean(x[np.where(grouping == g)[0], :], axis=0)
            for g in np.unique(grouping)
        ]))
        if link_fn is None:
            return groupwise_means
        else:
            return link_fn(groupwise_means)

    linker_groupwise_means, mu, rmsd, rank, s = groupwise_solve_lm(
        dmat=dmat,
        apply_fun=apply_fun,
        constraints=constraints
    )
    if inv_link_fn is not None:
        return inv_link_fn(linker_groupwise_means), mu, rmsd
    else:
        return linker_groupwise_means, mu, rmsd


def closedform_glm_scale(
        x: Union[np.ndarray, scipy.sparse.csr_matrix],
        design_scale: np.ndarray,
        constraints=None,
        size_factors=None,
        groupwise_means=None,
        link_fn=None,
        inv_link_fn=None,
        compute_scales_fun=None
):
    r"""
    Calculates a closed-form solution for the scale parameters of GLMs.
    :param x: The sample data
    :param design_scale: design matrix for scale
    :param constraints: some design constraints
    :param size_factors: size factors for X
    :param groupwise_means: optional, in case if already computed this can be specified to spare double-calculation
    :return: tuple (groupwise_scales, logphi, rmsd)
    """
    if size_factors is not None:
        x = x / size_factors

    # to circumvent nonlocal error
    provided_groupwise_means = groupwise_means

    def apply_fun(grouping):
        # Calculate group-wise means if not supplied. These are required for variance and MME computation.
        if provided_groupwise_means is None:
            gw_means = np.asarray(np.vstack([
                np.mean(x[np.where(grouping == g)[0], :], axis=0)
                for g in np.unique(grouping)
            ]))
        else:
            gw_means = provided_groupwise_means

        # calculated variance via E(x)^2 or directly depending on whether `mu` was specified
        if isinstance(x, scipy.sparse.csr_matrix):
            expect_xsq = np.asarray(np.vstack([
                np.asarray(np.mean(x[np.where(grouping == g)[0], :].power(2), axis=0))
                for g in np.unique(grouping)]
            ))
        else:
            expect_xsq = np.vstack([np.mean(np.square(x[np.where(grouping == g)[0], :]), axis=0)
                                    for g in np.unique(grouping)])
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
        dmat=design_scale,
        apply_fun=apply_fun,
        constraints=constraints
    )
    if inv_link_fn is not None:
        return inv_link_fn(linker_groupwise_scales), scaleparam, rmsd
    else:
        return linker_groupwise_scales, scaleparam, rmsd
