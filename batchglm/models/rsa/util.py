from typing import Union, Tuple, List

import patsy
import pandas as pd
import xarray as xr
import numpy as np

import batchglm.data as data_utils
from batchglm.utils.numeric import combine_matrices


def mixture_model_setup(
        sample_description: Union[np.ndarray, pd.DataFrame],
        formula,
        mixture_grouping: Union[str, np.ndarray],
        differing_factors: Union[Tuple[str], List[str]] = ("Intercept",),
        dim_names: Tuple[str, str] = ("design_params", "mixtures", "design_mixture_params")
) -> Tuple[patsy.highlevel.DesignMatrix, xr.DataArray, np.ndarray]:
    r"""
    Build a mixture description.

    :param sample_description: Some sample description which can be used to build a design matrix
    :param formula: Formula to set up a model design
    :param mixture_grouping: either string pointing to one column in `sample_description` or a list of assignments.

        Describes some (assumed) mixture assignment.
        The number of mixtures is derived from the number of unique elements in this grouping.
    :param differing_factors: Terms of the formula which are allowed to differ across mixtures
    :param dim_names: dimension names; defaults to ("mixtures", "design_params").
    :return: tuple: (design matrix, mixture design, initial mixture weights)
    """
    dmat: patsy.highlevel.DesignMatrix = data_utils.design_matrix(sample_description, formula=formula)

    # get differing and equal parameters
    differing_params = list()
    for param_slice in differing_factors:
        for param in dmat.design_info.column_names[dmat.design_info.term_name_slices[param_slice]]:
            differing_params.append(param)

    if isinstance(mixture_grouping, str):
        mixture_grouping = sample_description[mixture_grouping]

    groups, init_mixture_assignment = np.unique(mixture_grouping, return_inverse=True)

    num_mixtures = np.size(groups)
    num_observations = len(dmat)
    # num_design_params = len(dmat.design_info.column_names)

    # build mixture description
    mdesc = pd.DataFrame(0, columns=dmat.design_info.column_names, index=range(num_mixtures))
    for f in differing_params:
        mdesc[f] = range(num_mixtures)

    design_tensor = design_tensor_from_mixture_description(
        mixture_description=mdesc,
        dims=dim_names
    )

    # build init_mixture_weights matrix
    init_mixture_weights = np.zeros(shape=[num_observations, num_mixtures])
    init_mixture_weights[np.arange(num_observations), init_mixture_assignment] = 1

    return dmat, design_tensor, init_mixture_weights


def design_tensor_from_mixture_description(
        mixture_description: Union[xr.DataArray, pd.DataFrame],
        dims=("design_params", "mixtures", "design_mixture_params")
):
    r"""
    This method allows to specify in detail, which of the parameters in a design matrix should be equal across
    multiple mixtures.

    For example, if `mixture_description` would look like the following data frame:
    ::

                 Intercept   batch condition
        mixtures
        0                0     'a'         0
        1                1     'a'         0
        2                2     'b'         0
        3                3     'b'         0

    Then, `intercept` would differ across all mixtures, `batch` would be equal in mixtures 0 and 1 as well as 1 and 3.
    `condition` would be equal in all mixtures.

    Technically, it converts a 2D mixture description of shape (mixtures, properties) into a 3D design matrix of shape
    (properties, mixtures, design_mixture_params) by creating from each column in `mixture_description`
    a (non-confounded) design matrix and combining this list of design matrices to one 3D matrix.

    :param mixture_description: 2D mixture description of shape (mixtures, properties)
    :param dims: names of the dimensions of the returned xr.DataArray
    :return: 3D xr.DataArray of shape `dims`
    """
    df: pd.DataFrame
    if isinstance(mixture_description, xr.DataArray):
        df = pd.DataFrame(
            data=mixture_description.values,
            index=mixture_description[mixture_description.dims[0]],
            columns=mixture_description[mixture_description.dims[1]]
        )
    else:
        df = mixture_description
    df = df.astype(str)

    list_of_dmatrices = [patsy.highlevel.dmatrix("~ 0 + col", {"col": data}) for col, data in df.items()]
    names = df.columns

    data = combine_matrices(list_of_dmatrices)
    combined_dmat = xr.DataArray(
        dims=dims,
        data=data,
        coords={
            dims[0]: np.asarray(names),
            dims[2]: np.arange(data.shape[2])  # explicitly set parameter id's
        }
    )

    return combined_dmat
