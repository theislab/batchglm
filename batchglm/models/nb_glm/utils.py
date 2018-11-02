import numpy as np
import xarray as xr

from batchglm.utils.linalg import groupwise_solve_lm
from batchglm.models.glm import closedform_glm_mean


def closedform_nb_glm_logmu(
        X: xr.DataArray,
        design_loc,
        constraints=None,
        size_factors=None,
):
    """
    Calculates a closed-form solution for the `mu` parameters of negative-binomial GLMs.

    :param X:
    :param design_loc:
    :param constraints:
    :param size_factors:
    :return:
    """
    return closedform_glm_mean(X, design_loc, constraints, size_factors, link_fn=np.log)


def closedform_nb_glm_logphi(
        X: xr.DataArray,
        design_scale: xr.DataArray,
        constraints=None,
        size_factors=None,
        mu=None,
        groupwise_means=None,
):
    """
    Calculates a closed-form solution for the log-scale parameters of negative-binomial GLMs.
    Based on the Method-of-Moments estimator.

    :param X:
    :param design_scale:
    :param constraints:
    :param size_factors:
    :param mu: optional, if there are for example different mu's per observation.

        Used to calculate `Xdiff = X - mu`.
    :param groupwise_means: optional, in case if already computed this can be specified to spare double-calculation
    :return:
    """
    if size_factors is not None:
        X = np.divide(X, size_factors)

    def apply_fun(grouping):
        grouped_X = X.assign_coords(group=((X.dims[0],), grouping))
        nonlocal groupwise_means
        nonlocal mu

        if mu is None:
            Xdiff = grouped_X - grouped_X.mean(dim="observations")
        else:
            Xdiff = grouped_X - mu

        variance = np.square(Xdiff).groupby("group").mean(dim="observations")

        if groupwise_means is None:
            groupwise_means = grouped_X.groupby("group").mean(dim="observations")

        denominator = np.fmax(variance - groupwise_means, 0)
        denominator = np.nextafter(0, 1, out=denominator.values,
                                   where=denominator == 0,
                                   dtype=denominator.dtype)
        groupwise_scales = np.asarray(np.square(groupwise_means) / denominator)
        # clipping
        # r = np_clip_param(r, "r")
        groupwise_scales = np.nextafter(0, 1, out=groupwise_scales,
                                        where=groupwise_scales == 0,
                                        dtype=groupwise_scales.dtype)
        groupwise_scales = np.fmin(groupwise_scales, np.finfo(groupwise_scales.dtype).max)

        return np.log(groupwise_scales)

    groupwise_scales, logphi, rmsd, rank, _ = groupwise_solve_lm(
        dmat=design_scale,
        apply_fun=apply_fun,
        constraints=constraints
    )

    return groupwise_scales, logphi, rmsd
