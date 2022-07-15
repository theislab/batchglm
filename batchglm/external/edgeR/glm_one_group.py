import logging

import numpy as np

logger = logging.get_logger(__name__)


def fit(
    data: np.ndarray,
    size_factors: np.ndarray,
    dispersion: float,
    weights: np.ndarray = 1.0,
    maxit: int = 50,
    tolerance: int = 1e-10,
    cur_beta: np.ndarray = None,
):
    """
    Setting up initial values for beta as the log of the mean of the ratio of counts to offsets.
    * This is the exact solution for the gamma distribution (which is the limit of the NB as
    * the dispersion goes to infinity. However, if cur_beta is not NA, then we assume it's good.
    """
    low_value = 10 ** -10
    low_mask = data > low_value
    # nonzero = np.any(low_mask, axis=0)
    if cur_beta is None:
        cur_beta = np.zeros(size_factors.shape[0], dtype=float)
        total_weight = np.zeros_like(cur_beta)

        cur_beta = np.sum(data / np.exp(size_factors) * weights * low_mask, axis=0)
        total_weight = weights * size_factors.shape[0]
        cur_beta = np.log(cur_beta / total_weight)

    # Skipping to a result for all-zero rows.
    # if (!nonzero) {
    # return std::make_pair(R_NegInf, true);

    # // Newton-Raphson iterations to converge to mean.
    has_converged = np.zeros(data.shape[1], dtype=bool)
    for _ in range(maxit):
        # dl = np.zeros(data.shape[1], dtype=float)
        # info = np.zeros_like(dl)
        mu = np.exp(cur_beta + size_factors)
        denominator = 1 + mu * dispersion
        dl = np.sum((data - mu) / denominator * weights, axis=0)
        info = np.sum(mu / denominator * weights, axis=0)
        step = dl / info
        step = np.where(has_converged, 0.0, step)
        cur_beta += step
        has_converged = np.abs(step) < tolerance
        if np.all(has_converged):
            break
    else:
        logger.warning("Maximum iterations exceeded.")

    return cur_beta, has_converged
