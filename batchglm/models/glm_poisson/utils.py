import logging
from typing import Callable, Optional, Tuple, Union

import dask
import numpy as np
import scipy.sparse

from .external import closedform_glm_mean

logger = logging.getLogger("batchglm")


def closedform_poisson_glm_loglam(
    x: Union[np.ndarray, scipy.sparse.csr_matrix, dask.array.core.Array],
    design_loc: Union[np.ndarray, dask.array.core.Array],
    constraints_loc: Union[np.ndarray, dask.array.core.Array],
    size_factors: Optional[np.ndarray] = None,
    link_fn: Callable = np.log,
    inv_link_fn: Callable = np.exp,
):
    r"""
    Calculates a closed-form solution for the `lam` parameters of poisson GLMs.

    :param x: The sample data
    :param design_loc: design matrix for location
    :param constraints_loc: tensor (all parameters x dependent parameters)
        Tensor that encodes how complete parameter set which includes dependent
        parameters arises from indepedent parameters: all = <constraints, indep>.
        This form of constraints is used in vector generalized linear models (VGLMs).
    :param size_factors: size factors for X
    :return: tuple: (groupwise_means, mu, rmsd)
    """
    return closedform_glm_mean(
        x=x,
        dmat=design_loc,
        constraints=constraints_loc,
        size_factors=size_factors,
        link_fn=link_fn,
        inv_link_fn=inv_link_fn,
    )


def init_par(model, init_location: str) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
    r"""
    standard:
    Only initialise intercept and keep other coefficients as zero.

    closed-form:
    Initialize with Maximum Likelihood / Maximum of Momentum estimators

    Idea:
    $$
        \theta &= f(x) \\
        \Rightarrow f^{-1}(\theta) &= x \\
            &= (D \cdot D^{+}) \cdot x \\
            &= D \cdot (D^{+} \cdot x) \\
            &= D \cdot x' = f^{-1}(\theta)
    $$
    """
    train_loc = False

    def auto_loc(dmat: Union[np.ndarray, dask.array.core.Array]) -> str:
        """
        Checks if dmat is one-hot encoded and returns 'closed_form' if so, else 'standard'

        :param dmat The design matrix to check.
        """
        unique_params = np.unique(dmat)
        if isinstance(unique_params, dask.array.core.Array):
            unique_params = unique_params.compute()
        if len(unique_params) == 2 and unique_params[0] == 0.0 and unique_params[1] == 1.0:
            return "closed_form"
        logger.warning(
            (
                "Cannot use 'closed_form' init for loc model: "
                "design_loc is not one-hot encoded. Falling back to standard initialization."
            )
        )
        return "standard"

    groupwise_means = None

    init_location_str = init_location.lower()
    # Chose option if auto was chosen
    if init_location_str == "auto":

        init_location_str = auto_loc(model.design_loc)

    if init_location_str == "closed_form":
        groupwise_means, init_theta_location, rmsd_a = closedform_poisson_glm_loglam(
            x=model.x,
            design_loc=model.design_loc,
            constraints_loc=model.constraints_loc,
            size_factors=model.size_factors,
            link_fn=lambda lam: np.log(lam + np.nextafter(0, 1, dtype=lam.dtype)),  # why the epsilon?
        )
        # train mu, if the closed-form solution is inaccurate
        train_loc = not (np.all(np.abs(rmsd_a) < 1e-20) or rmsd_a.size == 0)
        if model.size_factors is not None:
            if np.any(model.size_factors != 1):
                train_loc = True

    elif init_location_str == "standard":
        overall_means = np.mean(model.x, axis=0)  # directly calculate the mean
        init_theta_location = np.zeros([model.num_loc_params, model.num_features])
        init_theta_location[0, :] = np.log(overall_means)
        train_loc = True
    elif init_location_str == "all_zero":
        init_theta_location = np.zeros([model.num_loc_params, model.num_features])
        train_loc = True
    else:
        raise ValueError("init_location string %s not recognized" % init_location)

    return init_theta_location, init_theta_location, train_loc, True
