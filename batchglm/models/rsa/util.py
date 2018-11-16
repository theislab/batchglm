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
        num_mixtures: int,
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


def mixture_prior_setup(
        mixture_grouping: Union[List, np.ndarray],
        mixture_model_desc: pd.DataFrame,
) -> pd.DataFrame:
    """
    Creates a mixture prior

    :param mixture_grouping: vector of length `num_observations` denoting the group of an observation.
    :param mixture_model_desc: Pandas DataFrame of shape `(num_groups, mixtures)` containing booleans.

        This table describes the structure of the mixture model:
        It allows to specify, which of the observations are allowed to be assigned to which mixture.
        Example:

        ..
            ```python3
            pd.DataFrame({
                "m_norm":  [1, 1, 1],
                "m_tr1.1": [0, 1, 0],
                "m_tr1.2": [0, 1, 0],
                "m_tr2.1": [0, 0, 1],
                "m_tr2.2": [0, 0, 1],
            }, index = ["norm", "tr1", "tr2"])
            ```

                  m_norm  m_tr1.1  m_tr1.2  m_tr2.1  m_tr2.2
            norm       1        0        0        0        0
            tr1        1        1        1        0        0
            tr2        1        0        0        1        1

        In this example, all normal samples are only allowed to occur in the mixture `m_norm`.
        All `tr1` samples can occur in the mixtures `m_norm`, `m_tr1.1` and `m_tr1.2`.
        All `tr2` samples can occur in the mixtures `m_norm`, `m_tr2.1` and `m_tr2.2`.
    :return: Pandas Dataframe of shape `(num_observations, num_mixtures)`
    """
    mixture_grouping = np.asarray(mixture_grouping)

    # num_mixtures = pin_mixtures
    # if pin_mixtures is not None:
    #     pin_mixtures = np.asarray(pin_mixtures)
    #     if np.issubdtype(pin_mixtures.dtype, np.integer):
    #         pinned_assignments = pin_mixtures
    #     else:
    #         if np.size(pin_mixtures) == 1:
    #             pin_mixtures = {pin_mixtures.flatten()[0]}
    #         else:
    #             pin_mixtures = set(pin_mixtures.astype(str))
    #         pinned_assignments = np.where(np.vectorize(lambda x: x in pin_mixtures)(mixture_grouping))
    #
    #     pin_mask = np.zeros([num_observations, num_mixtures])
    #     pin_mask[pinned_assignments] = 1
    #     pin_mask = pin_mask.astype(bool)
    #
    #     mixture_weight_priors = np.where(pin_mask, init_mixture_weights, np.ones_like(init_mixture_weights))
    # else:
    #     mixture_weight_priors = None

    if np.issubdtype(mixture_grouping.dtype, np.integer):
        return mixture_model_desc.iloc[mixture_grouping]
    else:
        return mixture_model_desc.loc[mixture_grouping]


# def mixture_weight_init(
#         mixture_priors: pd.DataFrame,
#         initial_mixture_assignments: pd.Series
# ):
#     """
#     Creates an array of initial mixture weights
#
#     :param mixture_priors: Pandas Dataframe of mixture priors. See also :func:mixture_prior_setup
#     :param initial_mixture_assignments: Pandas Series of length
#     :return: numpy array `(num_observations, num_mixtures)`
#     """
#     bcast_init_assignments = initial_mixture_assignments.loc[mixture_priors.index]
#
#     initial_mixture_weights = np.zeros(mixture_priors.shape)
#     initial_mixture_weights[range(len(mixture_priors)), bcast_init_assignments] = 1
#
#     return initial_mixture_weights


def mixture_model_setup(
        sample_description: Union[np.ndarray, pd.DataFrame],
        formula_loc,
        formula_scale,
        mixture_model_desc: pd.DataFrame,
        mixture_grouping: Union[str, Iterable],
        differing_factors_loc: Union[Tuple[str], List[str]] = ("Intercept",),
        differing_factors_scale: Union[Tuple[str], List[str]] = ("Intercept",),
        dim_names_loc: Tuple[str, str] = ("design_loc_params", "mixtures", "design_mixture_loc_params"),
        dim_names_scale: Tuple[str, str] = ("design_scale_params", "mixtures", "design_mixture_scale_params"),
) -> Tuple[
    patsy.highlevel.DesignMatrix,
    xr.DataArray,
    patsy.highlevel.DesignMatrix,
    xr.DataArray,
    pd.DataFrame
]:
    r"""
    Set up a mixture model.

    :param sample_description: Some sample description which can be used to build a design matrix
    :param formula_loc: Formula to set up a model design for location
    :param formula_scale: Formula to set up a model design for scale
    :param mixture_model_desc: Pandas DataFrame of shape `(num_groups, mixtures)` containing booleans.

        This table describes the structure of the mixture model:
        It allows to specify, which of the observations are allowed to be assigned to which mixture.
        Example:

        ..
            ```python3
            pd.DataFrame({
                "m_norm":  [1, 1, 1],
                "m_tr1.1": [0, 1, 0],
                "m_tr1.2": [0, 1, 0],
                "m_tr2.1": [0, 0, 1],
                "m_tr2.2": [0, 0, 1],
            }, index = ["norm", "tr1", "tr2"])
            ```

                  m_norm  m_tr1.1  m_tr1.2  m_tr2.1  m_tr2.2
            norm       1        0        0        0        0
            tr1        1        1        1        0        0
            tr2        1        0        0        1        1

        In this example, all normal samples are only allowed to occur in the mixture `m_norm`.
        All `tr1` samples can occur in the mixtures `m_norm`, `m_tr1.1` and `m_tr1.2`.
        All `tr2` samples can occur in the mixtures `m_norm`, `m_tr2.1` and `m_tr2.2`.
    :param mixture_grouping: either string pointing to one column in `sample_description` or a list of assignments.

        Describes which observations correspond to which group (i.e. row) in `mixture_model_desc`
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
            - mixture weight priors (`None` if `pin_mixtures` is not specified`)
    """
    num_mixtures = mixture_model_desc.shape[1]
    # num_observations = len(sample_description)

    if isinstance(mixture_grouping, str):
        mixture_grouping = sample_description[mixture_grouping]

    dmat_loc, design_tensor_loc = _mdesc_setup(
        sample_description=sample_description,
        formula=formula_loc,
        num_mixtures=num_mixtures,
        differing_factors=differing_factors_loc,
        dim_names=dim_names_loc
    )
    dmat_scale, design_tensor_scale = _mdesc_setup(
        sample_description=sample_description,
        formula=formula_scale,
        num_mixtures=num_mixtures,
        differing_factors=differing_factors_scale,
        dim_names=dim_names_scale
    )

    # num_design_params = len(dmat.design_info.column_names)
    mixture_weight_priors = mixture_prior_setup(
        mixture_grouping=mixture_grouping,
        mixture_model_desc=mixture_model_desc
    )
    # # build init_mixture_weights matrix
    # init_mixture_weights = np.zeros(shape=[num_observations, num_mixtures])
    # init_mixture_weights[np.arange(num_observations), init_mixture_assignment] = 1
    #
    # if pin_mixtures is not None:
    #     pin_mixtures = np.asarray(pin_mixtures)
    #     if np.issubdtype(pin_mixtures.dtype, np.integer):
    #         pinned_assignments = pin_mixtures
    #     else:
    #         if np.size(pin_mixtures) == 1:
    #             pin_mixtures = {pin_mixtures.flatten()[0]}
    #         else:
    #             pin_mixtures = set(pin_mixtures.astype(str))
    #         pinned_assignments = np.where(np.vectorize(lambda x: x in pin_mixtures)(mixture_grouping))
    #
    #     pin_mask = np.zeros([num_observations, num_mixtures])
    #     pin_mask[pinned_assignments] = 1
    #     pin_mask = pin_mask.astype(bool)
    #
    #     mixture_weight_priors = np.where(pin_mask, init_mixture_weights, np.ones_like(init_mixture_weights))
    # else:
    #     mixture_weight_priors = None

    return dmat_loc, design_tensor_loc, dmat_scale, design_tensor_scale, mixture_weight_priors


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
