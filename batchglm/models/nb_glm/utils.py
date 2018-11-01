import numpy as np
import xarray as xr

from batchglm.utils.linalg import groupwise_solve_lm


def closed_form_negbin_logmu(
        X: xr.DataArray,
        design_loc,
        constraints=None,
        size_factors=None
):
    if size_factors is not None:
        X = np.divide(X, size_factors)

    def apply_fun(grouping):
        grouped_data = X.assign_coords(group=((X.dims[0],), grouping))
        groupwise_means = grouped_data.groupby("group").mean(dim="observations").values
        # clipping
        groupwise_means = np.nextafter(0, 1, out=groupwise_means, where=groupwise_means == 0,
                                       dtype=groupwise_means.dtype)

        return np.log(groupwise_means)

    groupwise_means, logmu, rmsd, rank, s = groupwise_solve_lm(
        dmat=design_loc,
        apply_fun=apply_fun,
        constraints=constraints
    )

    return groupwise_means, logmu, rmsd


def closed_form_negbin_logphi(
        X: xr.DataArray,
        a: xr.DataArray,
        design_loc: xr.DataArray,
        design_scale: xr.DataArray,
        constraints=None,
        size_factors=None,
        groupwise_means=None,
):
    if size_factors is not None:
        X = np.divide(X, size_factors)

    def apply_fun(grouping):
        grouped_data = X.assign_coords(group=((X.dims[0],), grouping))
        nonlocal groupwise_means

        Xdiff = grouped_data - np.exp(design_loc.dot(a))
        variance = np.square(Xdiff).groupby("group").mean(dim="observations")

        if groupwise_means is None:
            groupwise_means = grouped_data.groupby("group").mean(dim="observations")

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
