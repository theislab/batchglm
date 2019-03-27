from typing import Union

import numpy as np
import xarray as xr

from .external import closedform_glm_mean
from .external import SparseXArrayDataArray


def closedform_bern_glm_logitmu(
        X: Union[xr.DataArray, SparseXArrayDataArray],
        design_loc,
        constraints_loc,
        size_factors=None,
        link_fn=lambda data: np.log(data/(1-data)),
        inv_link_fn=lambda data: 1/(1+np.exp(-data))
):
    r"""
    Calculates a closed-form solution for the `mu` parameters of bernoulli GLMs.

    :param X: The sample data
    :param design_loc: design matrix for location
    :param constraints_loc: tensor (all parameters x dependent parameters)
        Tensor that encodes how complete parameter set which includes dependent
        parameters arises from indepedent parameters: all = <constraints, indep>.
        This form of constraints is used in vector generalized linear models (VGLMs).
    :param size_factors: size factors for X
    :return: tuple: (groupwise_means, mu, rmsd)
    """
    return closedform_glm_mean(
        X=X,
        dmat=design_loc,
        constraints=constraints_loc,
        size_factors=size_factors,
        link_fn=link_fn,
        inv_link_fn=inv_link_fn
    )