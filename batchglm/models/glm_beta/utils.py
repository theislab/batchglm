from typing import Union

import numpy as np
import xarray as xr

from .external import closedform_glm_mean, closedform_glm_scale
from .external import SparseXArrayDataArray


def closedform_beta_glm_logmu(
        X: Union[xr.DataArray, SparseXArrayDataArray],
        design_loc,
        constraints_loc,
        design_scale: xr.DataArray,
        constraints=None,
        size_factors=None,
        link_fn=np.log,
        inv_link_fn=np.exp
):
    r"""
    Calculates a closed-form solution for the `mu` parameters of negative-binomial GLMs.

    :param X: The sample data
    :param design_loc: design matrix for location
    :param constraints_loc: tensor (all parameters x dependent parameters)
        Tensor that encodes how complete parameter set which includes dependent
        parameters arises from indepedent parameters: all = <constraints, indep>.
        This form of constraints is used in vector generalized linear models (VGLMs).
    :param size_factors: size factors for X
    :return: tuple: (groupwise_means, mu, rmsd)
    """
    groupwise_means, mu, rmsd1 =  closedform_glm_mean(
        X=X,
        dmat=design_loc,
        constraints=constraints_loc,
        size_factors=size_factors,
        link_fn=link_fn,
        inv_link_fn=inv_link_fn
    )

    groupwise_scale, var, rmsd2 =  closedform_glm_scale(
        X=X,
        design_scale=design_scale,
        constraints=constraints,
        size_factors=size_factors,
        groupwise_means=groupwise_means,
        link_fn=link_fn,
        compute_scales_fun=None
    )

    mu = mu / var * (mu * (1-mu) - var)
    return groupwise_means, mu, rmsd1


def closedform_beta_glm_logphi(
        X: Union[xr.DataArray, SparseXArrayDataArray],
        design_loc,
        constraints_loc,
        design_scale: xr.DataArray,
        constraints=None,
        size_factors=None,
        link_fn=np.log,
        inv_link_fn=np.exp,
):
    r"""
    Calculates a closed-form solution for the `mu` parameters of negative-binomial GLMs.

    :param X: The sample data
    :param design_loc: design matrix for location
    :param constraints_loc: tensor (all parameters x dependent parameters)
        Tensor that encodes how complete parameter set which includes dependent
        parameters arises from indepedent parameters: all = <constraints, indep>.
        This form of constraints is used in vector generalized linear models (VGLMs).
    :param size_factors: size factors for X
    :return: tuple: (groupwise_means, mu, rmsd)
    """
    groupwise_means, mu, rmsd1 = closedform_glm_mean(
        X=X,
        dmat=design_loc,
        constraints=constraints_loc,
        size_factors=size_factors,
        link_fn=link_fn,
        inv_link_fn=inv_link_fn
    )

    groupwise_scale, var, rmsd2 = closedform_glm_scale(
        X=X,
        design_scale=design_scale,
        constraints=constraints,
        size_factors=size_factors,
        groupwise_means=groupwise_means,
        link_fn=link_fn,
        compute_scales_fun=None,
    )

    var = (1 - mu) / var * (mu * (1 - mu) - var)
    return groupwise_scale, var, rmsd2
