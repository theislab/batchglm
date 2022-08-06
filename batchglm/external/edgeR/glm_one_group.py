import logging
from typing import Optional, Union

import numpy as np

from .external import BaseModelContainer

low_value = 1e-10
logger = logging.getLogger(__name__)


def get_single_group_start(
    x: np.ndarray,
    sf: Optional[np.ndarray] = None,
    weights: Optional[Union[np.ndarray, float]] = None,
):
    if weights is None:
        weights = np.ones_like(x)
    if weights.shape != x.shape:
        raise ValueError("Shape of weights must be idential to shape of model.x")

    total_weights = weights.sum(axis=0)

    if sf is None:
        sf = np.log(1.0)

    theta_location = np.sum(np.where(x > low_value, x / np.exp(sf) * weights, 0), axis=0, keepdims=True)
    theta_location = np.log(theta_location / total_weights)
    return theta_location


def fit_single_group(
    model: BaseModelContainer,
    maxit: int = 50,
    tolerance: int = 1e-10,
):
    """
    Setting up initial values for beta as the log of the mean of the ratio of counts to offsets.
    * This is the exact solution for the gamma distribution (which is the limit of the NB as
    * the dispersion goes to infinity. However, if cur_beta is not NA, then we assume it's good.
    """
    low_mask = np.all(model.x <= low_value, axis=0).compute()
    unconverged_idx = np.where(~low_mask)[0]

    iteration = 0
    weights = 1.0

    step = np.zeros((1, model.num_features), dtype=float)

    while iteration < maxit:
        loc_j = model.location_j(unconverged_idx)
        scale_j = 1 / model.scale_j(unconverged_idx)
        denominator = 1 + loc_j * scale_j

        dl = np.sum((model.x[:, unconverged_idx] - loc_j) / denominator * weights, axis=0).compute()

        info = np.sum(loc_j / denominator * weights, axis=0).compute()
        cur_step = dl / info
        step[0, unconverged_idx] = cur_step
        model.theta_location = model.theta_location.compute() + step
        unconverged_idx = unconverged_idx[np.abs(cur_step) >= tolerance]
        if len(unconverged_idx) == 0:
            break
        step.fill(0)
        iteration += 1
    else:
        logger.warning("Maximum iterations exceeded.")
