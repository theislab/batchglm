import dask
import logging
import numpy as np
import scipy.sparse
from typing import Union

from .external import closedform_glm_mean, closedform_glm_scale


def closedform_nb_glm_logmu(
        x: Union[np.ndarray, scipy.sparse.csr_matrix],
        design_loc: np.ndarray,
        constraints_loc,
        size_factors=None,
        link_fn=np.log,
        inv_link_fn=np.exp
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
        inv_link_fn=inv_link_fn
    )


def closedform_nb_glm_logphi(
        x: Union[np.ndarray, scipy.sparse.csr_matrix],
        design_scale: np.ndarray,
        constraints=None,
        size_factors=None,
        groupwise_means=None,
        link_fn=np.log,
        invlink_fn=np.exp
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
        compute_scales_fun=compute_scales_fun
    )


def init_par(
        input_data,
        init_a,
        init_b,
        init_model
):
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
        init_a_str = None
        if isinstance(init_a, str):
            init_a_str = init_a.lower()
            # Chose option if auto was chosen
            if init_a.lower() == "auto":
                if isinstance(input_data.design_loc, dask.array.core.Array):
                    dloc = input_data.design_loc.compute()
                else:
                    dloc = input_data.design_loc
                one_hot = len(np.unique(dloc)) == 2 and \
                    np.abs(np.min(dloc) - 0.) == 0. and \
                    np.abs(np.max(dloc) - 1.) == 0.
                init_a = "standard" if not one_hot else "closed_form"

            if init_a.lower() == "closed_form":
                groupwise_means, init_a, rmsd_a = closedform_nb_glm_logmu(
                    x=input_data.x,
                    design_loc=input_data.design_loc,
                    constraints_loc=input_data.constraints_loc,
                    size_factors=input_data.size_factors,
                    link_fn=lambda mu: np.log(mu+np.nextafter(0, 1, dtype=mu.dtype))
                )

                # train mu, if the closed-form solution is inaccurate
                train_loc = not (np.all(np.abs(rmsd_a) < 1e-20) or rmsd_a.size == 0)

                if input_data.size_factors is not None:
                    if np.any(input_data.size_factors != 1):
                        train_loc = True
            elif init_a.lower() == "standard":
                overall_means = np.mean(input_data.x, axis=0)  # directly calculate the mean
                init_a = np.zeros([input_data.num_loc_params, input_data.num_features])
                init_a[0, :] = np.log(overall_means)
                train_loc = True
            elif init_a.lower() == "all_zero":
                init_a = np.zeros([input_data.num_loc_params, input_data.num_features])
                train_loc = True
            else:
                raise ValueError("init_a string %s not recognized" % init_a)

        if isinstance(init_b, str):
            if init_b.lower() == "auto":
                init_b = "standard"

            if init_b.lower() == "standard":
                groupwise_scales, init_b_intercept, rmsd_b = closedform_nb_glm_logphi(
                    x=input_data.x,
                    design_scale=input_data.design_scale[:, [0]],
                    constraints=input_data.constraints_scale[[0], :][:, [0]],
                    size_factors=input_data.size_factors,
                    groupwise_means=None,
                    link_fn=lambda r: np.log(r+np.nextafter(0, 1, dtype=r.dtype))
                )
                init_b = np.zeros([input_data.num_scale_params, input_data.num_features])
                init_b[0, :] = init_b_intercept
            elif init_b.lower() == "closed_form":
                dmats_unequal = False
                if input_data.design_loc.shape[1] == input_data.design_scale.shape[1]:
                    if np.any(input_data.design_loc != input_data.design_scale):
                        dmats_unequal = True

                inits_unequal = False
                if init_a_str is not None:
                    if init_a_str != init_b:
                        inits_unequal = True

                if inits_unequal or dmats_unequal:
                    raise ValueError("cannot use closed_form init for scale model " +
                                     "if scale model differs from loc model")

                groupwise_scales, init_b, rmsd_b = closedform_nb_glm_logphi(
                    x=input_data.x,
                    design_scale=input_data.design_scale,
                    constraints=input_data.constraints_scale,
                    size_factors=input_data.size_factors,
                    groupwise_means=groupwise_means,
                    link_fn=lambda r: np.log(r)
                )
            elif init_b.lower() == "all_zero":
                init_b = np.zeros([input_data.num_scale_params, input_data.x.shape[1]])
            else:
                raise ValueError("init_b string %s not recognized" % init_b)
    else:
        # Locations model:
        if isinstance(init_a, str) and (init_a.lower() == "auto" or init_a.lower() == "init_model"):
            my_loc_names = set(input_data.loc_names)
            my_loc_names = my_loc_names.intersection(set(init_model.input_data.loc_names))

            init_loc = np.zeros([input_data.num_loc_params, input_data.num_features])
            for parm in my_loc_names:
                init_idx = np.where(init_model.input_data.loc_names == parm)[0]
                my_idx = np.where(input_data.loc_names == parm)[0]
                init_loc[my_idx] = init_model.a_var[init_idx]

            init_a = init_loc
            logging.getLogger("batchglm").debug("Using initialization based on input model for mean")

        # Scale model:
        if isinstance(init_b, str) and (init_b.lower() == "auto" or init_b.lower() == "init_model"):
            my_scale_names = set(input_data.scale_names)
            my_scale_names = my_scale_names.intersection(init_model.input_data.scale_names)

            init_scale = np.zeros([input_data.num_scale_params, input_data.num_features])
            for parm in my_scale_names:
                init_idx = np.where(init_model.input_data.scale_names == parm)[0]
                my_idx = np.where(input_data.scale_names == parm)[0]
                init_scale[my_idx] = init_model.b_var[init_idx]

            init_b = init_scale
            logging.getLogger("batchglm").debug("Using initialization based on input model for dispersion")

    return init_a, init_b, train_loc, train_scale

