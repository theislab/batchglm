import time

import numpy as np

from .external import BaseModelContainer


def nb_deviance(model: BaseModelContainer, idx=...):

    """
    Python version of the method implemented in a C++ function in edgeR.
    """

    eps = 1e-8
    eps2 = 1e-4

    y = model.x[:, idx].compute().copy()
    mu = model.location[:, idx].compute().copy()
    phi = 1 / model.scale[0, idx].compute()

    y += eps
    mu += eps

    if isinstance(phi, float):
        phi = np.full(y.shape[1], phi)

    deviance = np.zeros_like(y, dtype=float)
    poisson_idx = phi < eps2  # .compute()
    if np.any(poisson_idx):
        deviance[:, poisson_idx] = _poisson_deviance(poisson_idx, y, mu, phi)

    non_poisson_idx = ~poisson_idx
    y_non_poisson = y[:, non_poisson_idx]
    mu_non_poisson = mu[:, non_poisson_idx]
    phi_non_poisson = phi[non_poisson_idx]
    product = mu_non_poisson * phi_non_poisson

    mask = product > 1e6

    deviance[:, non_poisson_idx] = np.where(
        mask,
        _gamma_deviance(y_non_poisson, mu_non_poisson, product),
        _nb_deviance(y_non_poisson, mu_non_poisson, phi_non_poisson),
    )
    return np.sum(deviance, axis=0)


def _poisson_deviance(idx, y, mu, phi):
    y_poisson = y[:, idx]
    mu_poisson = mu[:, idx]
    phi_poisson = phi
    resid = y_poisson - mu_poisson
    return 2 * (
        y_poisson * np.log(y_poisson / mu_poisson)
        - resid
        - 0.5 * resid * phi_poisson * (1 + phi_poisson * (2 / 3 * resid - y))
    )


def _gamma_deviance(y, mu, product):
    return 2 * ((y - mu) / mu - np.log(y / mu)) * mu / (1 + product)


def _nb_deviance(y, mu, phi):
    inv_phi = 1 / phi
    return 2 * (y * np.log(y / mu) + (y + inv_phi) * np.log((mu + inv_phi) / (y + inv_phi)))
