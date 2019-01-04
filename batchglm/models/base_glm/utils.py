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
from .external import weighted_mean, weighted_variance


def parse_design(
        data: Union[anndata.AnnData, xr.Dataset] = None,
        design_matrix: Union[pd.DataFrame, patsy.design_info.DesignMatrix, xr.DataArray] = None,
        coords: dict = None,
        design_key: str = "design",
        dims=("observations", "design_params")
) -> xr.DataArray:
    r"""
    Parser for design matrices.

    :param data: some dataset possibly containing a design matrix
    :param design_matrix: some design matrix
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


def closedform_glm_mean(
        X: xr.DataArray,
        dmat,
        constraints=None,
        size_factors=None,
        weights=None,
        link_fn: Union[callable, None] = None
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
    :param weights: the weights of the arrays' elements; if `none` it will be ignored.
    :param link_fn: linker function for GLM
    :return: tuple: (groupwise_means, mu, rmsd)
    """
    if size_factors is not None:
        X = np.divide(X, size_factors)

    def apply_fun(grouping):
        grouped_data = X.assign_coords(group=((X.dims[0],), grouping)).groupby("group")

        if weights is None:
            groupwise_means = grouped_data.mean(X.dims[0]).values
        else:
            grouped_weights = xr.DataArray(
                data=weights,
                dims=(X.dims[0],),
                coords={
                    "group": ((X.dims[0],), grouping),
                }
            ).groupby("group")

            groupwise_means: xr.DataArray = xr.concat([
                weighted_mean(
                    d,
                    weights=w,
                    axis=0
                ) for (g, d), (g, w) in zip(grouped_data, grouped_weights)
            ], dim="group")
            groupwise_means = groupwise_means.values

        if link_fn is None:
            return groupwise_means
        else:
            return link_fn(groupwise_means)

    groupwise_means, mu, rmsd, rank, s = groupwise_solve_lm(
        dmat=dmat,
        apply_fun=apply_fun,
        constraints=constraints
    )

    return groupwise_means, mu, rmsd


def closedform_glm_var(
        X: xr.DataArray,
        dmat,
        constraints=None,
        size_factors=None,
        weights=None,
        link_fn: Union[callable, None] = None
):
    """
    Calculates a closed-form solution for the variance parameters of GLMs.

    :param X: The input data array
    :param dmat: some design matrix
    :param constraints: constraints
    :param size_factors: size factors for X
    :param weights: the weights of the arrays' elements; if `none` it will be ignored.
    :param link_fn: linker function for GLM
    :return: tuple: (groupwise_variance, phi, rmsd)
    """
    if size_factors is not None:
        X = np.divide(X, size_factors)

    def apply_fun(grouping):
        grouped_data = X.assign_coords(group=((X.dims[0],), grouping))
        if weights is None:
            groupwise_variance = grouped_data.var(X.dims[0]).values
        else:
            grouped_weights = xr.DataArray(
                data=weights,
                dims=(X.dims[0],),
                coords={
                    "group": ((X.dims[0],), grouping),
                }
            ).groupby("group")

            groupwise_variance: xr.DataArray = xr.concat([
                weighted_variance(d, w, axis=0) for (g, d), (g, w) in zip(grouped_data, grouped_weights)
            ], dim="group")
            groupwise_variance = groupwise_variance.values

        if link_fn is None:
            return groupwise_variance
        else:
            return link_fn(groupwise_variance)

    groupwise_variance, phi, rmsd, rank, s = groupwise_solve_lm(
        dmat=dmat,
        apply_fun=apply_fun,
        constraints=constraints
    )

    return groupwise_variance, phi, rmsd
