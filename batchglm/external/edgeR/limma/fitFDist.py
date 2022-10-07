import logging
from typing import Optional, Tuple

import numpy as np
import patsy
import scipy.special

from .effects import calc_effects

logger = logging.getLogger(__name__)


def fit_f_dist(x: np.ndarray, df1: np.ndarray, covariate: Optional[np.ndarray]):
    """
    Moment estimation of the parameters of a scaled F-distribution.
    The numerator degrees of freedom is given, the scale factor and denominator df is to be estimated.
    This function is a python version of limma's fitFDist function.
    """
    # Check x
    n = len(x)
    if n == 1:
        return x, 0

    ok = np.isfinite(df1) & (df1 > 1e-15)

    # Check covariate

    if covariate is None:
        spline_df = 1
    else:
        assert len(x) == len(df1) == len(covariate), "All inputs must have the same length"
        if np.any(np.isnan(covariate)):
            raise ValueError("NA covariate values are not allowed.")
        isfin = np.isfinite(covariate)
        if not np.all(isfin):
            if np.any(isfin):
                min_covariate = np.min(covariate)
                max_covariate = np.max(covariate)
                np.clip(covariate, min_covariate, max_covariate, out=covariate)
            else:
                covariate = np.sign(covariate)

    # Remove missing or infinite or negative values and zero degrees of freedom
    ok &= np.isfinite(x) & (x > -1e-15)
    n_ok = np.sum(ok)
    if n_ok == 1:
        return x[ok], 0
    not_all_ok = n_ok < n
    if not_all_ok:
        x = x[ok]
        df1 = df1[ok]
        if covariate is not None:
            covariate_not_ok = covariate[~ok]
            covariate = covariate[ok]

    # Set df for spline trend
    if covariate is not None:
        spline_df = 1 + int(n_ok >= 3) + int(n_ok >= 6) + int(n_ok >= 30)
        spline_df = np.minimum(spline_df, len(np.unique(covariate)))
        # If covariate takes only one unique value or insufficient
        # observations, recall with NULL covariate
        if spline_df < 2:
            scale, df2 = fit_f_dist(x=x, df1=df1, covariate=None)
            scale = np.full(n, scale)
            return scale, df2

    # Avoid exactly zero values
    x = np.maximum(x, 0)
    m = np.median(x)
    if m == 0:
        logger.warning("More than half of residual variances are exactly zero: eBayes unreliable")
        m = 1
    elif np.any(x == 0):
        logger.warning("Zero sample variances detected, have been offset away from zero")
    x = np.maximum(x, 1e-5 * m)

    # Better to work on with log(F)
    z = np.log(x)
    e = z - scipy.special.digamma(df1 / 2) + np.log(df1 / 2)

    if covariate is None:
        e_mean = np.mean(e)
        e_var = np.sum(np.square(e - e_mean), keepdims=True) / (n_ok - 1)
    else:
        # formula = f"bs(x, df={spline_df}, degree=3, include_intercept=False)"
        # formula = f"cr(x, df={spline_df})-1"
        formula = f"cr(x, df={spline_df}) -1"

        design = patsy.dmatrix(formula, {"x": covariate})

        loc_params, _, rank, _ = scipy.linalg.lstsq(design, e)
        if not_all_ok:
            design2 = patsy.build_design_matrices([design.design_info], data={"x": covariate_not_ok})[0]

            e_mean = np.zeros(n, dtype=float)

            e_mean[ok] = np.matmul(design, loc_params)
            e_mean[~ok] = np.matmul(design2, loc_params)
        else:
            e_mean = np.matmul(design, loc_params)

        (qr, tau), r = scipy.linalg.qr(np.asarray(design), mode="raw")
        effects = calc_effects(qr, tau, e)
        e_var = np.mean(np.square(effects[rank:]), keepdims=True)

    # Estimate scale and df2
    e_var = e_var - np.mean(scipy.special.polygamma(x=df1 / 2, n=1))  # this is the trigamma function in R

    # return 0, e_var
    e_var = np.array([0.5343055])
    if e_var > 0:
        df2 = 2 * trigamma_inverse(e_var)
        np.save(arr=e_mean, file="/home/mario/phd/collabs/batchglm/eman.csv")
        s20 = np.exp(e_mean + scipy.special.digamma(df2 / 2) - np.log(df2 / 2))
    else:
        df2 = np.array([np.inf])
        if covariate is None:
            """
            Use simple pooled variance, which is MLE of the scale in this case.
            Versions of limma before Jan 2017 returned the limiting
            value of the evar>0 estimate, which is larger.
            """
            s20 = np.mean(x)
        else:
            s20 = np.exp(e_mean)
    return s20, df2


def trigamma_inverse(x: np.ndarray):
    """
    Solve trigamma(y) = x for y. Python version of limma's trigammaInverse function.
    """
    # Non-numeric or zero length input
    if len(x) == 0:
        return 0

    # Treat out-of-range values as special cases
    omit = np.isnan(x)
    if np.any(omit):
        y = x
        if np.any(~omit):
            y[~omit] = trigamma_inverse(x[~omit])
        return y

    omit = x < 0
    if np.any(omit):
        y = x
        y[omit] = np.nan
        logger.warning("NaNs produced")
        if np.any(~omit):
            y[~omit] = trigamma_inverse(x[~omit])
        return y

    omit = x > 1e7
    if np.any(omit):
        y = x
        y[omit] = 1 / np.sqrt(x[omit])
        if np.any(~omit):
            y[~omit] = trigamma_inverse(x[~omit])
        return y

    omit = x < 1e-6
    if np.any(omit):
        y = x
        y[omit] = 1 / x[omit]
        if np.any(~omit):
            y[~omit] = trigamma_inverse(x[~omit])
        return y
    """
    Newton's method
    1/trigamma(y) is convex, nearly linear and strictly > y-0.5,
    so iteration to solve 1/x = 1/trigamma is monotonically convergent
    """
    y = 0.5 + 1 / x
    for _ in range(50):
        tri = scipy.special.polygamma(x=y, n=1)  # this is the trigamma function (psi^1(x))
        dif = tri * (1 - tri / x) / scipy.special.polygamma(x=y, n=2)  # this is psi^2(x)
        y = y + dif
        if np.max(-dif / y) < 1e-8:
            break
    else:
        logger.warning("Iteration limit exceeded for trigammaInverse function.")

    return y


def fit_f_dist_robustly(
    var: np.ndarray, df1: np.ndarray, winsor_tail_p: Tuple[float, float], covariate: Optional[np.ndarray] = None
):
    pass
