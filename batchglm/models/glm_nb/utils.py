import logging
from typing import Callable, Optional, Union

import dask
import numpy as np
import scipy.sparse

from .external import closedform_glm_mean, closedform_glm_scale


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


def init_par(input_data, init_location, init_scale, init_model):
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
    train_loc = True
    train_scale = True

    if init_model is None:
        groupwise_means = None
        init_location_str = None
        if isinstance(init_location, str):
            init_location_str = init_location.lower()
            # Chose option if auto was chosen
            if init_location.lower() == "auto":
                if isinstance(input_data.design_loc, dask.array.core.Array):
                    dloc = input_data.design_loc.compute()
                else:
                    dloc = input_data.design_loc
                one_hot = (
                    len(np.unique(dloc)) == 2
                    and np.abs(np.min(dloc) - 0.0) == 0.0
                    and np.abs(np.max(dloc) - 1.0) == 0.0
                )
                init_location = "standard" if not one_hot else "closed_form"

            if init_location.lower() == "closed_form":
                groupwise_means, init_location, rmsd_a = closedform_nb_glm_logmu(
                    x=input_data.x,
                    design_loc=input_data.design_loc,
                    constraints_loc=input_data.constraints_loc,
                    size_factors=input_data.size_factors,
                    link_fn=lambda mu: np.log(mu + np.nextafter(0, 1, dtype=mu.dtype)),
                )

                # train mu, if the closed-form solution is inaccurate
                train_loc = not (np.all(np.abs(rmsd_a) < 1e-20) or rmsd_a.size == 0)

                if input_data.size_factors is not None:
                    if np.any(input_data.size_factors != 1):
                        train_loc = True
            elif init_location.lower() == "standard":
                overall_means = np.mean(input_data.x, axis=0)  # directly calculate the mean
                init_location = np.zeros([input_data.num_loc_params, input_data.num_features])
                init_location[0, :] = np.log(overall_means)
                train_loc = True
            elif init_location.lower() == "all_zero":
                init_location = np.zeros([input_data.num_loc_params, input_data.num_features])
                train_loc = True
            else:
                raise ValueError("init_location string %s not recognized" % init_location)

        if isinstance(init_scale, str):
            if init_scale.lower() == "auto":
                init_scale = "standard"

            if init_scale.lower() == "standard":
                groupwise_scales, init_scale_intercept, rmsd_b = closedform_nb_glm_logphi(
                    x=input_data.x,
                    design_scale=input_data.design_scale[:, [0]],
                    constraints=input_data.constraints_scale[[0], :][:, [0]],
                    size_factors=input_data.size_factors,
                    groupwise_means=None,
                    link_fn=lambda r: np.log(r + np.nextafter(0, 1, dtype=r.dtype)),
                )
                init_scale = np.zeros([input_data.num_scale_params, input_data.num_features])
                init_scale[0, :] = init_scale_intercept
            elif init_scale.lower() == "closed_form":
                dmats_unequal = False
                if input_data.design_loc.shape[1] == input_data.design_scale.shape[1]:
                    if np.any(input_data.design_loc != input_data.design_scale):
                        dmats_unequal = True

                inits_unequal = False
                if init_location_str is not None:
                    if init_location_str != init_scale:
                        inits_unequal = True

                if inits_unequal or dmats_unequal:
                    raise ValueError(
                        "cannot use closed_form init for scale model " + "if scale model differs from loc model"
                    )

                groupwise_scales, init_scale, rmsd_b = closedform_nb_glm_logphi(
                    x=input_data.x,
                    design_scale=input_data.design_scale,
                    constraints=input_data.constraints_scale,
                    size_factors=input_data.size_factors,
                    groupwise_means=groupwise_means,
                    link_fn=lambda r: np.log(r),
                )
            elif init_scale.lower() == "all_zero":
                init_scale = np.zeros([input_data.num_scale_params, input_data.x.shape[1]])
            else:
                raise ValueError("init_scale string %s not recognized" % init_scale)
    else:
        # Locations model:
        if isinstance(init_location, str) and (
            init_location.lower() == "auto" or init_location.lower() == "init_model"
        ):
            my_loc_names = set(input_data.loc_names)
            my_loc_names = my_loc_names.intersection(set(init_model.input_data.loc_names))

            init_loc = np.zeros([input_data.num_loc_params, input_data.num_features])
            for parm in my_loc_names:
                init_idx = np.where(init_model.input_data.loc_names == parm)[0]
                my_idx = np.where(input_data.loc_names == parm)[0]
                init_loc[my_idx] = init_model.theta_location[init_idx]

            init_location = init_loc
            logging.getLogger("batchglm").debug("Using initialization based on input model for mean")

        # Scale model:
        if isinstance(init_scale, str) and (init_scale.lower() == "auto" or init_scale.lower() == "init_model"):
            my_scale_names = set(input_data.scale_names)
            my_scale_names = my_scale_names.intersection(init_model.input_data.scale_names)

            init_scale = np.zeros([input_data.num_scale_params, input_data.num_features])
            for parm in my_scale_names:
                init_idx = np.where(init_model.input_data.scale_names == parm)[0]
                my_idx = np.where(input_data.scale_names == parm)[0]
                init_scale[my_idx] = init_model.theta_scale[init_idx]

            init_scale = init_scale
            logging.getLogger("batchglm").debug("Using initialization based on input model for dispersion")

    return init_location, init_scale, train_loc, train_scale
