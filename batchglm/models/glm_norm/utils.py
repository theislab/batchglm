import logging
from typing import Tuple, Union

import dask
import numpy as np
import scipy.sparse

from .external import closedform_glm_scale

logger = logging.getLogger("batchglm")


def closedform_norm_glm_logsd(
    x: Union[np.ndarray, scipy.sparse.csr_matrix, dask.array.core.Array],
    design_scale: Union[np.ndarray, dask.array.core.Array],
    constraints=None,
    size_factors=None,
    groupwise_means=None,
    link_fn=np.log,
):
    r"""
    Calculates a closed-form solution for the log-scale parameters of normal GLMs.

    :param x: The sample data
    :param design_scale: design matrix for scale
    :param constraints: some design constraints
    :param size_factors: size factors for X
    :param groupwise_means: optional, in case if already computed this can be specified to spare double-calculation
    :return: tuple (groupwise_scales, logsd, rmsd)
    """

    def compute_scales_fun(variance, mean):
        groupwise_scales = np.sqrt(variance)
        return groupwise_scales

    return closedform_glm_scale(
        x=x,
        design_scale=design_scale,
        constraints=constraints,
        size_factors=size_factors,
        groupwise_means=groupwise_means,
        link_fn=link_fn,
        compute_scales_fun=compute_scales_fun,
    )


def init_par(model, init_location: str, init_scale: str) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
    r"""
    standard:
    Only initialise intercept and keep other coefficients as zero.

    closed-form:
    Initialize with Maximum Likelihood / Maximum of Momentum estimators

    Idea:
    $$
        \theta &= f(x) \\
        \Rightarrow f^{-1}(\theta) &= x \\
            &= (D \cdot D^{+}) \cdot x \\
            &= D \cdot (D^{+} \cdot x) \\
            &= D \cdot x' = f^{-1}(\theta)
    $$
    """
    groupwise_means = None

    init_location_str = init_location.lower()
    # Chose option if auto was chosen
    auto_or_closed_form = init_location_str == "auto" or init_location_str == "closed_form"
    if auto_or_closed_form or init_location_str == "all_zero":
        if auto_or_closed_form:
            logger.warning(
                (
                    "There is no need for closed form location model initialization"
                    "because it is already closed form - falling back to zeros"
                )
            )
        init_theta_location = np.zeros([model.num_loc_params, model.num_features])
    elif init_location_str == "standard":
        overall_means = np.mean(model.x, axis=0)  # directly calculate the mean
        init_theta_location = np.zeros([model.num_loc_params, model.num_features])
        init_theta_location[0, :] = np.log(overall_means)
    else:
        raise ValueError("init_location string %s not recognized" % init_location)

    init_scale_str = init_scale.lower()
    if init_scale_str == "auto":
        init_scale_str = "standard"

    if init_scale_str == "standard":
        groupwise_scales, init_scale_intercept, rmsd_b = closedform_norm_glm_logsd(
            x=model.x,
            design_scale=model.design_scale[:, [0]],
            constraints=model.constraints_scale[[0], :][:, [0]],
            size_factors=model.size_factors,
            groupwise_means=None,
            link_fn=lambda r: np.log(r + np.nextafter(0, 1, dtype=r.dtype)),
        )
        init_theta_scale = np.zeros([model.num_scale_params, model.num_features])
        init_theta_scale[0, :] = init_scale_intercept
    elif init_scale_str == "closed_form":
        groupwise_scales, init_theta_scale, rmsd_b = closedform_norm_glm_logsd(
            x=model.x,
            design_scale=model.design_scale,
            constraints=model.constraints_scale,
            size_factors=model.size_factors,
            groupwise_means=groupwise_means,
        )
    elif init_scale_str == "all_zero":
        init_theta_scale = np.zeros([model.num_scale_params, model.x.shape[1]])
    else:
        raise ValueError("init_scale string %s not recognized" % init_scale_str)

    return init_theta_location, init_theta_scale, True, True
