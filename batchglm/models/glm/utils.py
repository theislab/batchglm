from typing import Union, Dict

try:
    import anndata
except ImportError:
    anndata = None
import xarray as xr
import numpy as np
import pandas as pd

import patsy


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
