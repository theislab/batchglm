from typing import Any

import numpy as np

from .maximizeInterpolant import maximizeInterpolant


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
    given log-likelihood values on a grid of theta values

    returns tuple span, overall prior, shared_loglik(trended prior), trend, individual
    """

    n_features, n_theta = loglik.shape
    #       Check covariate and trend
    if covariate is None:
        trend_method = "none"

    #       Set span
    if span is None:
        if n_features < 50:
            out_span = 1
        else:
            out_span = 0.25 + 0.75 * np.sqrt(50 / n_features)

    #       overall prior
    if overall:
        out_overall = maximizeInterpolant(theta, np.sum(loglik, axis=0))
    else:
        out_overall = None

    #       trended prior
    if m0 is None:
        if trend_method == "none":
            m0 = np.broadcast_to(np.sum(loglik, axis=0), loglik.shape)
        elif trend_method == "loess":
            m0 = loess_by_col(loglik, covariate, span=out_span)
        else:
            raise NotImplementedError(f"Method {trend_method} is not yet implemented.")

    if trend:
        out_trend = maximizeInterpolant(theta, m0)
    else:
        out_trend = None

    #       weighted empirical Bayes posterior estimates
    if individual:
        assert np.all(np.isfinite(prior_n)), "prior_n must not contain infinite values."
        l0a = loglik + prior_n * m0
        out_individual = maximizeInterpolant(theta, l0a)

    return out_span, out_overall, m0, out_trend, out_individual


def loess_by_col():
    pass
