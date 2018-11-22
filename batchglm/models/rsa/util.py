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


def mixture_constraint_setup(
        mixture_grouping: Union[List, np.ndarray],
        mixture_model_desc: pd.DataFrame,
) -> pd.DataFrame:
    """
    Creates a mixture constraint

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

    if np.issubdtype(mixture_grouping.dtype, np.integer):
        return mixture_model_desc.iloc[mixture_grouping]
    else:
        return mixture_model_desc.loc[mixture_grouping]


def mixture_model_setup(
        sample_description: Union[np.ndarray, pd.DataFrame],
        formula_loc,
        formula_scale,
        mixture_model_desc: pd.DataFrame,
        mixture_grouping: Union[str, Iterable],
        init_mixture_assignments: pd.DataFrame = None,
        differing_factors_loc: Union[Tuple[str], List[str]] = ("Intercept",),
        differing_factors_scale: Union[Tuple[str], List[str]] = ("Intercept",),
        dim_names_loc: Tuple[str, str] = ("design_loc_params", "mixtures", "design_mixture_loc_params"),
        dim_names_scale: Tuple[str, str] = ("design_scale_params", "mixtures", "design_mixture_scale_params"),
) -> Tuple[
    patsy.highlevel.DesignMatrix,
    xr.DataArray,
    patsy.highlevel.DesignMatrix,
    xr.DataArray,
    pd.DataFrame,
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
    :param init_mixture_assignments: Pandas DataFrame with same rows and columns as `mixture_model_desc`.
        Allows to define which samples should be initially assigned to which mixtures.
        All positive float values are valid weightings.
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
            - mixture weight constraints
            - initial mixture weights
    """
    num_mixtures = mixture_model_desc.shape[1]
    # num_observations = len(sample_description)

    if isinstance(mixture_grouping, str):
        mixture_grouping = np.asarray(sample_description[mixture_grouping])

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
    mixture_weight_constraints = mixture_constraint_setup(
        mixture_grouping=mixture_grouping,
        mixture_model_desc=mixture_model_desc
    )

    # create initial mixture weights
    if init_mixture_assignments is not None:
        if np.issubdtype(mixture_grouping.dtype, np.integer):
            init_mixture_weights = init_mixture_assignments.iloc[mixture_grouping]
        else:
            init_mixture_weights = init_mixture_assignments.loc[mixture_grouping]
    else:
        init_mixture_weights = mixture_weight_constraints

    return dmat_loc, design_tensor_loc, dmat_scale, design_tensor_scale, mixture_weight_constraints, init_mixture_weights


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


def plot_mixture_weights(
        mixture_prob: Union[np.ndarray, xr.DataArray],
        mixture_labels: List[str] = None,
        group_assignments: List[str] = None,
        rasterized=True,
        cbar_name="group",
        cmap="vlag",
        title="\nmixture probabilities of observations",
        xlab="mixtures",
        ylab="observations",
        **kwargs
):
    r"""
    Simplifies plotting mixture weights in a heatmap

    :param mixture_prob: mixture probabilities of shape (observations, mixtures).
    :param mixture_labels: optional labels of mixtures; can also be passed by mixture_prob.coords["mixtures"]
    :param group_assignments: optional list of strings assigning each observation to a group;
        can also be passed by mixture_prob.coords["mixture_group"]
    :param cbar_name: name of the color bar
    :param cmap: color map to be used for the heatmap
    :param rasterized: should the heatmap be rasterized? See also seaborn.heatmap()
    :param kwargs: other arguments which will be passed to seaborn.clustermap()
    :param title: title of the plot
    :param xlab: x-axis label
    :param ylab: y-axis label
    :return: seaborn.clustermap()
    """
    # import matplotlib.pyplot as plt
    import seaborn as sns

    if not isinstance(mixture_prob, xr.DataArray):
        mixture_prob = xr.DataArray(
            dims=("observations", "mixtures"),
            data=mixture_prob
        )

    if mixture_labels is None:
        mixture_labels = mixture_prob.coords[mixture_prob.dims[-1]].values

    if group_assignments is not None:
        mixture_prob.coords["mixture_group"] = xr.DataArray(
            dims=mixture_prob.dims[0],
            data=group_assignments
        )

    if "mixture_group" in mixture_prob.coords:
        mixture_prob = mixture_prob.sortby("mixture_group")

        group_assignments = mixture_prob.coords["mixture_group"].values

    plot_data = mixture_prob
    plot_df = pd.DataFrame(plot_data.values, columns=mixture_labels)

    if group_assignments is not None:
        cbar_labels = pd.Series(group_assignments, name=cbar_name)
        cbar_pal = sns.color_palette("colorblind", np.unique(cbar_labels).size)
        cbar_lut = dict(zip(map(str, np.unique(cbar_labels)), cbar_pal))
        cbar_colors = pd.Series(cbar_labels, index=plot_df.index).map(cbar_lut)

        kwargs["row_colors"] = cbar_colors

    g = sns.clustermap(
        plot_df,
        rasterized=rasterized,
        xticklabels=mixture_labels,
        row_cluster=False,
        col_cluster=False,
        cmap=cmap,
        vmin=0,
        vmax=1,
        **kwargs
    )

    if group_assignments is not None:
        for label in np.unique(cbar_labels):
            g.ax_col_dendrogram.bar(0, 0, color=cbar_lut[label], label=label, linewidth=0)

        lg = g.ax_col_dendrogram.legend(title=cbar_name, loc="lower center", ncol=6)

    g.cax.set_position([.20, .2, .03, .45])
    g.ax_col_dendrogram.set_title(title)
    g.ax_heatmap.set_xlabel(xlab)
    g.ax_heatmap.set_ylabel(ylab)

    return g
