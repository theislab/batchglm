from typing import Union

try:
    import anndata
except ImportError:
    anndata = None

import xarray as xr
import numpy as np
import pandas as pd

import patsy

from .external import groupwise_solve_lm
from .external import weighted_mean
from .external import SparseXArrayDataArray


def parse_design(
        data: Union[anndata.AnnData, xr.Dataset] = None,
        design_matrix: Union[pd.DataFrame, patsy.design_info.DesignMatrix, xr.DataArray] = None,
        coords: dict = None,
        design_key: str = "design",
        dims=("observations", "design_params")
) -> xr.DataArray:
    r"""
    Parser for design matrices.

    :param data: Dataset possibly containing a design matrix.
    :param design_matrix: Design matrix.
    :param coords: dict containing coords which should be added to the returned xr.DataArray
    :param design_key: key where `data` possibly contains a design matrix (data[design_key] ?)
    :param dims: tuple containing names for (<row>, <column>)
    :return: xr.DataArray containing the design matrix
    """
    if design_matrix is not None:
        if isinstance(design_matrix, patsy.design_info.DesignMatrix):
            dmat = xr.DataArray(design_matrix, dims=dims)
            dmat.coords[dims[1]] = design_matrix.design_info.column_names
        elif isinstance(design_matrix, xr.DataArray):
            dmat = design_matrix
            dmat = dmat.rename({
                dmat.dims[i]: d for i, d in enumerate(dims)
            })
        elif isinstance(design_matrix, pd.DataFrame):
            dmat = xr.DataArray(np.asarray(design_matrix), dims=dims)
            dmat.coords[dims[1]] = design_matrix.columns
        else:
            dmat = xr.DataArray(design_matrix, dims=dims)
    elif anndata is not None and isinstance(data, anndata.AnnData):
        dmat = data.obsm[design_key]
        dmat = xr.DataArray(dmat, dims=dims)
    elif isinstance(data, xr.Dataset):
        dmat: xr.DataArray = data[design_key]
        dmat = dmat.rename({
            dmat.dims[i]: d for i, d in enumerate(dims)
        })
    else:
        raise ValueError("Missing %s matrix!" % design_key)

    if coords is not None:
        dmat.coords.merge(coords)
        # dmat.coords[dims[1]] = names
    elif dims[1] not in dmat.coords:
        # ### add dmat.coords[dim] = 0..len(dim) if dmat.coords[dim] is non-existent and `names` was not provided.
        # Note that `dmat.coords[dim]` returns a corresponding index array although dmat.coords[dim] is not set.
        # However, other ways accessing this coordinates will raise errors instead;
        # therefore, it is necessary to set this index explicitly
        dmat.coords[dims[1]] = dmat.coords[dims[1]]
        # raise ValueError("Could not find names for %s; Please specify them manually." % dim)

    return dmat


def parse_constraints(
        dmat: xr.Dataset,
        dims,
        constraints: np.ndarray = None,
        constraint_par_names: list = None
) -> xr.DataArray:
    r"""
    Parser for constraint matrices.

    :param dmat: Design matrix.
    :param constraints: Constraint matrix.
    :param constraint_par_names: list of coords for xr.DataArray
    :param dims: tuple containing names for (design_params, params) = (all parameters, independent parameters)
    :return: xr.DataArray containing the constraint matrix
    """
    if constraints is None:
        constraints = np.identity(n=dmat.shape[1])
        # Use given parameter names if constraint matrix is identity.
        par_names = dmat.coords[dims[0]]
    else:
        # Cannot use given parameter names if constraint matrix is not identity: Make up new ones.
        par_names = ["var_"+str(x) for x in range(constraints.shape[1])]
        assert constraints.shape[0] == dmat.shape[1], "constraint dimension mismatch"

    constraints_mat = xr.DataArray(
        dims=dims,
        data=constraints
    )
    constraints_mat.coords[dims[0]] = dmat.coords[dims[0]]
    if constraint_par_names is None:
        constraint_par_names = par_names

    constraints_mat.coords[dims[1]] = constraint_par_names

    return constraints_mat


def closedform_glm_mean(
        X: Union[xr.DataArray, SparseXArrayDataArray],
        dmat,
        constraints=None,
        size_factors=None,
        link_fn: Union[callable, None] = None,
        inv_link_fn: Union[callable, None] = None
):
    r"""
    Calculates a closed-form solution for the mean parameters of GLMs.

    :param X: The input data array
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
        if isinstance(X, SparseXArrayDataArray):
            X = X.multiply(np.ones_like(size_factors) / size_factors, copy=True)
        else:
            X = np.divide(X, size_factors)

    def apply_fun(grouping):
        if isinstance(X, SparseXArrayDataArray):
            X.assign_coords(coords=("group", grouping))
            X.groupby("group")
        else:
            grouped_data = X.assign_coords(group=((X.dims[0],), grouping)).groupby("group")

        if isinstance(X, SparseXArrayDataArray):
            groupwise_means = X.group_means(X.dims[0])
        else:
            groupwise_means = grouped_data.mean(X.dims[0]).values

        if link_fn is None:
            return groupwise_means
        else:
            return link_fn(groupwise_means)

    linker_groupwise_means, mu, rmsd, rank, s = groupwise_solve_lm(
        dmat=dmat,
        apply_fun=apply_fun,
        constraints=constraints
    )

    return inv_link_fn(linker_groupwise_means), mu, rmsd


def closedform_glm_scale(
        X: Union[xr.DataArray, SparseXArrayDataArray],
        design_scale: xr.DataArray,
        constraints=None,
        size_factors=None,
        groupwise_means=None,
        link_fn=None,
        compute_scales_fun=None
):
    r"""
    Calculates a closed-form solution for the scale parameters of GLMs.

    :param X: The sample data
    :param design_scale: design matrix for scale
    :param constraints: some design constraints
    :param size_factors: size factors for X
    :param groupwise_means: optional, in case if already computed this can be specified to spare double-calculation
    :return: tuple (groupwise_scales, logphi, rmsd)
    """
    if size_factors is not None:
        if isinstance(X, SparseXArrayDataArray):
            X = X.multiply(np.ones_like(size_factors) / size_factors, copy=True)
        else:
            X = np.divide(X, size_factors)

    # to circumvent nonlocal error
    provided_groupwise_means = groupwise_means

    def apply_fun(grouping):
        if isinstance(X, SparseXArrayDataArray):
            X.assign_coords(coords=("group", grouping))
            X.groupby("group")
        else:
            grouped_data = X.assign_coords(group=((X.dims[0],), grouping))

        # Calculate group-wise means if not supplied. These are required for variance and MME computation.
        if provided_groupwise_means is None:
            if isinstance(X, SparseXArrayDataArray):
                gw_means = X.group_means(X.dims[0])
            else:
                gw_means = grouped_data.groupby("group").mean(X.dims[0]).values
        else:
            if isinstance(X, SparseXArrayDataArray):
                X._group_means = provided_groupwise_means
            gw_means = provided_groupwise_means

        # calculated variance via E(x)^2 or directly depending on whether `mu` was specified
        if isinstance(X, SparseXArrayDataArray):
            variance = X.group_vars(X.dims[0])
        else:
            expect_xsq = np.square(grouped_data).groupby("group").mean(X.dims[0])
            expect_x_sq = np.square(gw_means)
            variance = expect_xsq - expect_x_sq

        if compute_scales_fun is not None:
            groupwise_scales = compute_scales_fun(variance, gw_means)
        else:
            groupwise_scales = variance

        # # clipping
        # # r = np_clip_param(r, "r")
        # groupwise_scales = np.nextafter(0, 1, out=groupwise_scales,
        #                                 where=groupwise_scales == 0,
        #                                 dtype=groupwise_scales.dtype)
        # groupwise_scales = np.fmin(groupwise_scales, np.finfo(groupwise_scales.dtype).max)
        if link_fn is not None:
            return link_fn(groupwise_scales)
        return groupwise_scales

    groupwise_scales, scaleparam, rmsd, rank, _ = groupwise_solve_lm(
        dmat=design_scale,
        apply_fun=apply_fun,
        constraints=constraints
    )

    return groupwise_scales, scaleparam, rmsd
