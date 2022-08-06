from typing import Any

import dask.array
import numpy as np

from .c_utils import loess_by_col
from .maximizeInterpolant import maximize_interpolant


def wleb(
    theta: Any,
    loglik: Any,
    prior_n: int = 5,
    covariate: np.ndarray = None,
    trend_method: str = "loess",
    span: Any = None,
    overall: bool = True,
    trend: bool = True,
    individual: bool = True,
    m0: Any = None,
):
    """
    Weighted likelihood empirical Bayes for estimating a parameter vector theta
    given log-likelihood values on a grid of theta values.
    The method is taken over from edgeR's wleb method.

    :param theta: the parameter vector which is a grid of disperions values.
    :param loglik: a log-likelihood matrix with shape (features, len(theta))
        containing the ll for each feature given a certain theta.
    :param prior_n: ???
    :param covariate: the average log counts per million for each feature over
        the observations
    :param trend_method: The method to use for calculating the trend. Currently,
        only loess as implemented in edgeR is supported.
    :param span: Optional window size used during the trend estimation.
    :param overall: If true, compute the overall prior ???
    :param trend: If true, compute the trend over all thetas
    :param individual: If true, compute weighted empirical bayes posterior estimates.
    :param m0: Optional output of a trend fitting procedure as specified
        by trend_method.

    :return: Tuple(out_span, out_overall, m0, out_trend, out_individual)
    """
    n_features, n_theta = loglik.shape
    if covariate is None:
        trend_method = "none"

    if span is None:
        if n_features < 50:
            span = 1
        else:
            span = 0.25 + 0.75 * np.sqrt(50 / n_features)
    out_span = span

    out_overall = None
    out_trend = None
    out_individual = None

    if overall:
        out_overall = maximize_interpolant(theta, np.sum(loglik, axis=0, keepdims=True))

    # calculate trended prior
    if m0 is None:
        if trend_method == "none":
            m0 = np.broadcast_to(np.sum(loglik, axis=0), loglik.shape)
        elif trend_method == "loess":
            m0, _ = loess(loglik, covariate, span=out_span)
        else:
            raise NotImplementedError(f"Method {trend_method} is not yet implemented.")

    if trend:
        out_trend = maximize_interpolant(theta, m0)

    # weighted empirical Bayes posterior estimates
    if individual:
        assert np.all(np.isfinite(prior_n)), "prior_n must not contain infinite values."
        l0a = loglik + prior_n * m0
        out_individual = maximize_interpolant(theta, l0a)

    return out_span, out_overall, m0, out_trend, out_individual


def loess(y: np.ndarray, x: np.ndarray, span: float):
    """
    Wrapper around loess as implemented in edgeR. This calls the C++
    function loess_by_col.
    """
    n_features = y.shape[0]
    if x is None:
        x = np.arange(n_features)
    if isinstance(x, dask.array.core.Array):
        x = x.compute()

    order = np.argsort(x, kind="stable")
    y = y[order]
    x = x[order]

    n_span = int(np.minimum(np.floor(span * n_features), n_features))

    if n_span <= 1:
        y_smooth = y
        leverages = np.arange(n_features)
        return y_smooth, leverages

    y_smooth = loess_by_col(x, y, n_span)
    y_smooth[order] = y_smooth.copy()
    leverages[order] = leverages.copy()

    return y_smooth, leverages
