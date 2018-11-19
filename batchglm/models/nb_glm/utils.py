from typing import Union

import numpy as np
import xarray as xr

from batchglm.utils.linalg import groupwise_solve_lm
from batchglm.utils.numeric import weighted_mean
from batchglm.models.glm import closedform_glm_mean


def closedform_nb_glm_logmu(
        X: xr.DataArray,
        design_loc,
        constraints=None,
        size_factors=None,
        weights=None,
        link_fn=np.log
):
    r"""
    Calculates a closed-form solution for the `mu` parameters of negative-binomial GLMs.

    :param X: The sample data
    :param design_loc: design matrix for location
    :param constraints: some design constraints
    :param size_factors: size factors for X
    :param weights: the weights of the arrays' elements; if `none` it will be ignored.
    :return: tuple: (groupwise_means, mu, rmsd)
    """
    return closedform_glm_mean(
        X=X,
        dmat=design_loc,
        constraints=constraints,
        size_factors=size_factors,
        weights=weights,
        link_fn=link_fn
    )


def closedform_nb_glm_logphi(
        X: xr.DataArray,
        design_scale: xr.DataArray,
        constraints=None,
        size_factors=None,
        weights: Union[np.ndarray, xr.DataArray] = None,
        mu=None,
        groupwise_means=None,
        link_fn=np.log
):
    r"""
    Calculates a closed-form solution for the log-scale parameters of negative-binomial GLMs.
    Based on the Method-of-Moments estimator.

    :param X: The sample data
    :param design_scale: design matrix for scale
    :param constraints: some design constraints
    :param size_factors: size factors for X
    :param weights: the weights of the arrays' elements; if `none` it will be ignored.
    :param mu: optional, if there are for example different mu's per observation.

        Used to calculate `Xdiff = X - mu`.
    :param groupwise_means: optional, in case if already computed this can be specified to spare double-calculation
    :return: tuple (groupwise_scales, logphi, rmsd)
    """
    if size_factors is not None:
        X = np.divide(X, size_factors)

    # to circumvent nonlocal error
    provided_groupwise_means = groupwise_means
    provided_weights = weights
    provided_mu = mu

    def apply_fun(grouping):
        grouped_X = X.assign_coords(group=((X.dims[0],), grouping))

        # convert weights into a xr.DataArray
        if provided_weights is not None:
            weights = xr.DataArray(
                data=provided_weights,
                dims=(X.dims[0],),
                coords={
                    "group": ((X.dims[0],), grouping),
                }
            )
        else:
            weights = None

        # calculate group-wise means if necessary
        if provided_groupwise_means is None:
            if weights is None:
                groupwise_means = grouped_X.mean(X.dims[0]).values
            else:
                # for each group: calculate weighted mean
                groupwise_means: xr.DataArray = xr.concat([
                    weighted_mean(d, w, axis=0) for (g, d), (g, w) in zip(
                        grouped_X.groupby("group"),
                        weights.groupby("group"))
                ], dim="group")
        else:
            groupwise_means = provided_groupwise_means

        # calculated (x - mean) depending on whether `mu` was specified
        if provided_mu is None:
            Xdiff = grouped_X - groupwise_means
        else:
            Xdiff = grouped_X - provided_mu

        if weights is None:
            # for each group:
            #   calculate mean of (X - mean)^2
            variance = np.square(Xdiff).groupby("group").mean(X.dims[0])
        else:
            # for each group:
            #   calculate weighted mean of (X - mean)^2
            variance: xr.DataArray = xr.concat([
                weighted_mean(d, w, axis=0) for (g, d), (g, w) in zip(
                    np.square(Xdiff).groupby("group"),
                    weights.groupby("group")
                )
            ], dim="group")

        denominator = np.fmax(variance - groupwise_means, np.sqrt(np.nextafter(0, 1, dtype=variance.dtype)))
        groupwise_scales = np.square(groupwise_means) / denominator

        # # clipping
        # # r = np_clip_param(r, "r")
        # groupwise_scales = np.nextafter(0, 1, out=groupwise_scales,
        #                                 where=groupwise_scales == 0,
        #                                 dtype=groupwise_scales.dtype)
        # groupwise_scales = np.fmin(groupwise_scales, np.finfo(groupwise_scales.dtype).max)

        return link_fn(groupwise_scales)

    groupwise_scales, logphi, rmsd, rank, _ = groupwise_solve_lm(
        dmat=design_scale,
        apply_fun=apply_fun,
        constraints=constraints
    )

    return groupwise_scales, logphi, rmsd
