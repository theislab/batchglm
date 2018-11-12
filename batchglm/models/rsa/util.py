from typing import Union, Tuple, List, Iterable

import patsy
import pandas as pd
import xarray as xr
import numpy as np

import batchglm.data as data_utils
from batchglm.utils.numeric import combine_matrices


def _mdesc_setup(
        sample_description: Union[np.ndarray, pd.DataFrame],
        formula,
        groups: Union[str, np.ndarray],
        differing_factors: Union[Tuple[str], List[str]] = ("Intercept",),
        dim_names: Tuple[str, str] = ("design_params", "mixtures", "design_mixture_params")
):
    dmat: patsy.highlevel.DesignMatrix = data_utils.design_matrix(sample_description, formula=formula)

    # get differing and equal parameters
    differing_params = list()
    for param_slice in differing_factors:
        for param in dmat.design_info.column_names[dmat.design_info.term_name_slices[param_slice]]:
            differing_params.append(param)

    # if isinstance(mixture_grouping, str):
    #     mixture_grouping = sample_description[mixture_grouping]
    #
    # groups, _ = np.unique(mixture_grouping.astype(str), return_inverse=True)

    num_mixtures = np.size(groups)
    # num_observations = len(sample_description)
    # num_design_params = len(dmat.design_info.column_names)

    # build mixture description
    mdesc = pd.DataFrame(0, columns=dmat.design_info.column_names, index=range(num_mixtures))
    for f in differing_params:
        mdesc[f] = range(num_mixtures)

    design_tensor = design_tensor_from_mixture_description(
        mixture_description=mdesc,
        dims=dim_names
    )

    return dmat, design_tensor


def mixture_model_setup(
        sample_description: Union[np.ndarray, pd.DataFrame],
        formula_loc,
        formula_scale,
        mixture_grouping: Union[str, np.ndarray],
        pin_mixtures: Union[Iterable[str], Iterable[int], None] = None,
        differing_factors_loc: Union[Tuple[str], List[str]] = ("Intercept",),
        differing_factors_scale: Union[Tuple[str], List[str]] = ("Intercept",),
        dim_names_loc: Tuple[str, str] = ("design_loc_params", "mixtures", "design_mixture_loc_params"),
        dim_names_scale: Tuple[str, str] = ("design_scale_params", "mixtures", "design_mixture_scale_params"),
) -> Tuple[
    patsy.highlevel.DesignMatrix,
    xr.DataArray,
    patsy.highlevel.DesignMatrix,
    xr.DataArray,
    np.ndarray,
    np.ndarray
]:
    r"""
    Build a mixture description.

    :param sample_description: Some sample description which can be used to build a design matrix
    :param formula_loc: Formula to set up a model design for location
    :param formula_scale: Formula to set up a model design for scale
    :param mixture_grouping: either string pointing to one column in `sample_description` or a list of assignments.

        Describes some (assumed) mixture assignment.
        The number of mixtures is derived from the number of unique elements in this grouping.
    :param pin_mixtures:
        Allows to specify, which of the observations should be assigned permanently to the initial mixture,
        i.e. are not allowed to change.

        Can be either:
            - Iterable of strings: Will fix the assignments where `mixture_grouping == anyof(pin_mixtures)`
            - Iterable of ints: Will fix the assignments at `observations[pin_mixtures]`
    :param differing_factors_loc: Terms of the location formula which are allowed to differ across mixtures
    :param differing_factors_scale: Terms of the scale formula which are allowed to differ across mixtures
    :param dim_names_loc: dimension names; defaults to ("mixtures", "design_params").
    :param dim_names_scale: dimension names; defaults to ("mixtures", "design_params").
    :return:
        tuple:
            - design_loc matrix
            - location mixture design
            - design_scale matrix
            - scale mixture design
            - initial mixture weights
            - mixture weight priors (`None` if `pin_mixtures` is not specified`)
    """

    if isinstance(mixture_grouping, str):
        mixture_grouping = sample_description[mixture_grouping]
    mixture_grouping = mixture_grouping.astype(str)

    groups, init_mixture_assignment = np.unique(mixture_grouping, return_inverse=True)

    dmat_loc, design_tensor_loc = _mdesc_setup(
        sample_description=sample_description,
        formula=formula_loc,
        groups=groups,
        differing_factors=differing_factors_loc,
        dim_names=dim_names_loc
    )
    dmat_scale, design_tensor_scale = _mdesc_setup(
        sample_description=sample_description,
        formula=formula_scale,
        groups=groups,
        differing_factors=differing_factors_scale,
        dim_names=dim_names_scale
    )

    num_mixtures = np.size(groups)
    num_observations = len(sample_description)
    # num_design_params = len(dmat.design_info.column_names)

    # build init_mixture_weights matrix
    init_mixture_weights = np.zeros(shape=[num_observations, num_mixtures])
    init_mixture_weights[np.arange(num_observations), init_mixture_assignment] = 1

    if pin_mixtures is not None:
        pin_mixtures = np.asarray(pin_mixtures)
        if np.issubdtype(pin_mixtures.dtype, np.integer):
            pinned_assignments = pin_mixtures
        else:
            if np.size(pin_mixtures) == 1:
                pin_mixtures = {pin_mixtures.flatten()[0]}
            else:
                pin_mixtures = set(pin_mixtures.astype(str))
            pinned_assignments = np.where(np.vectorize(lambda x: x in pin_mixtures)(mixture_grouping))

        pin_mask = np.zeros([num_observations, num_mixtures])
        pin_mask[pinned_assignments] = 1
        pin_mask = pin_mask.astype(bool)

        mixture_weight_priors = np.where(pin_mask, init_mixture_weights, np.ones_like(init_mixture_weights))
    else:
        mixture_weight_priors = None

    return dmat_loc, design_tensor_loc, dmat_scale, design_tensor_scale, init_mixture_weights, mixture_weight_priors


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
