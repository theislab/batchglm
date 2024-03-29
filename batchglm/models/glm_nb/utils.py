import logging
from typing import Callable, Optional, Tuple, Union

import dask
import numpy as np
import scipy.sparse

from .external import closedform_glm_mean, closedform_glm_scale

logger = logging.getLogger("batchglm")


def closedform_nb_glm_logmu(
    x: Union[np.ndarray, scipy.sparse.csr_matrix, dask.array.core.Array],
    design_loc: Union[np.ndarray, dask.array.core.Array],
    constraints_loc: Union[np.ndarray, dask.array.core.Array],
    size_factors: Optional[np.ndarray] = None,
    link_fn: Callable = np.log,
    inv_link_fn: Callable = np.exp,
):
    r"""
    Calculates a closed-form solution for the `mu` parameters of negative-binomial GLMs.

    :param x: The sample data
    :param design_loc: design matrix for location
    :param constraints_loc: tensor (all parameters x dependent parameters)
        Tensor that encodes how complete parameter set which includes dependent
        parameters arises from indepedent parameters: all = <constraints, indep>.
        This form of constraints is used in vector generalized linear models (VGLMs).
    :param size_factors: size factors for X
    :return: tuple: (groupwise_means, mu, rmsd)
    """
    return closedform_glm_mean(
        x=x,
        dmat=design_loc,
        constraints=constraints_loc,
        size_factors=size_factors,
        link_fn=link_fn,
        inv_link_fn=inv_link_fn,
    )


def closedform_nb_glm_logphi(
    x: Union[np.ndarray, scipy.sparse.csr_matrix, dask.array.core.Array],
    design_scale: Union[np.ndarray, dask.array.core.Array],
    constraints: Optional[Union[np.ndarray, dask.array.core.Array]] = None,
    size_factors: Optional[np.ndarray] = None,
    groupwise_means: Optional[np.ndarray] = None,
    link_fn: Callable = np.log,
    invlink_fn: Callable = np.exp,
):
    r"""
    Calculates a closed-form solution for the log-scale parameters of negative-binomial GLMs.
    Based on the Method-of-Moments estimator.

    :param x: The sample data
    :param design_scale: design matrix for scale
    :param constraints: some design constraints
    :param size_factors: size factors for X
    :param groupwise_means: optional, in case if already computed this can be specified to spare double-calculation
    :return: tuple (groupwise_scales, logphi, rmsd)
    """

    def compute_scales_fun(variance, mean):
        denominator = np.fmax(variance - mean, np.sqrt(np.nextafter(0, 1, dtype=variance.dtype)))
        groupwise_scales = np.square(mean) / denominator
        return groupwise_scales

    return closedform_glm_scale(
        x=x,
        design_scale=design_scale,
        constraints=constraints,
        size_factors=size_factors,
        groupwise_means=groupwise_means,
        link_fn=link_fn,
        inv_link_fn=invlink_fn,
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
    train_loc = False

    def auto_loc(dmat: Union[np.ndarray, dask.array.core.Array]) -> str:
        """
        Checks if dmat is one-hot encoded and returns 'closed_form' if so, else 'standard'

        :param dmat The design matrix to check.
        """
        unique_params = np.unique(dmat)
        if isinstance(unique_params, dask.array.core.Array):
            unique_params = unique_params.compute()
        if len(unique_params) == 2 and unique_params[0] == 0.0 and unique_params[1] == 1.0:
            return "closed_form"
        logger.warning(
            (
                "Cannot use 'closed_form' init for loc model: "
                "design_loc is not one-hot encoded. Falling back to standard initialization."
            )
        )
        return "standard"

    groupwise_means = None

    init_location_str = init_location.lower()
    # Chose option if auto was chosen
    if init_location_str == "auto":

        init_location_str = auto_loc(model.design_loc)

    if init_location_str == "closed_form":
        groupwise_means, init_theta_location, rmsd_a = closedform_nb_glm_logmu(
            x=model.x,
            design_loc=model.design_loc,
            constraints_loc=model.constraints_loc,
            size_factors=model.size_factors,
            link_fn=lambda mu: np.log(mu + np.nextafter(0, 1, dtype=mu.dtype)),
        )
        # train mu, if the closed-form solution is inaccurate
        train_loc = not (np.all(np.abs(rmsd_a) < 1e-20) or rmsd_a.size == 0)
        if model.size_factors is not None:
            if np.any(model.size_factors != 1):
                train_loc = True

    elif init_location_str == "standard":
        overall_means = np.mean(model.x, axis=0)  # directly calculate the mean
        init_theta_location = np.zeros([model.num_loc_params, model.num_features])
        init_theta_location[0, :] = np.log(overall_means)
        train_loc = True
    elif init_location_str == "all_zero":
        init_theta_location = np.zeros([model.num_loc_params, model.num_features])
        train_loc = True
    else:
        raise ValueError("init_location string %s not recognized" % init_location)

    init_scale_str = init_scale.lower()
    if init_scale_str == "auto":
        init_scale_str = "standard"

    if init_scale_str == "standard":
        groupwise_scales, init_scale_intercept, rmsd_b = closedform_nb_glm_logphi(
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
        if not np.array_equal(model.design_loc, model.design_scale):
            raise ValueError("Cannot use 'closed_form' init for scale model: design_scale != design_loc.")
        if init_location_str is not None and init_location_str != init_scale_str:
            raise ValueError(
                "Cannot use 'closed_form' init for scale model: init_location != 'closed_form' which is required."
            )

        groupwise_scales, init_theta_scale, rmsd_b = closedform_nb_glm_logphi(
            x=model.x,
            design_scale=model.design_scale,
            constraints=model.constraints_scale,
            size_factors=model.size_factors,
            groupwise_means=groupwise_means,
            link_fn=lambda r: np.log(r),
        )
    elif init_scale_str == "all_zero":
        init_theta_scale = np.zeros([model.num_scale_params, model.x.shape[1]])
    else:
        raise ValueError("init_scale string %s not recognized" % init_scale_str)

    return init_theta_location, init_theta_scale, train_loc, True
