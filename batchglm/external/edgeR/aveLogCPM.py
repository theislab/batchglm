import numpy as np

from .glm_one_group import fit


def avg_log_cpm(
    counts,
    size_factors,
    prior_count: int = 2,
    dispersion: np.ndarray = None,
    weights: np.ndarray = None,
    maxit=50,
    tolerance=1e-10,
):
    #       Compute average log2-cpm for each gene over all samples.
    #       This measure is designed to be used as the x-axis for all abundance-dependent trend analyses in edgeR.
    #       It is generally held fixed through an edgeR analysis.
    #       Original author: Gordon Smyth
    #       Created 25 Aug 2012. Last modified 19 Nov 2018.

    #       Check dispersion
    if dispersion is None:
        dispersion = 0.05

    #   Check weights
    if weights is None:
        weights = 1.0

    #   Calling the C++ code

    adjusted_prior, adjusted_size_factors = add_priors(prior_count, size_factors)
    # return adjusted_prior, adjusted_size_factors
    x = np.array(counts, dtype=float)  # model.x.copy()
    x += adjusted_prior
    output = fit(
        data=x,
        size_factors=adjusted_size_factors,
        dispersion=dispersion,
        weights=weights,
        maxit=maxit,
        tolerance=tolerance,
    )
    output = (output + np.log(1e6)) / np.log(2)

    return output


def add_priors(prior_count: int, size_factors: np.ndarray):

    factors = np.exp(size_factors)
    avg_factors = np.mean(factors)
    adjusted_priors = prior_count * factors / avg_factors

    adjusted_size_factors = np.log(factors + 2 * adjusted_priors)

    return adjusted_priors, adjusted_size_factors
