from typing import Union

import numpy as np
import xarray as xr

from .external import closedform_glm_mean, closedform_glm_scale
from .external import SparseXArrayDataArray


def closedform_beta_glm_logp(
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
    Calculates a closed-form solution for the `p` parameters of beta GLMs.

    :param X: The sample data
    :param design_loc: design matrix for location
    :param constraints_loc: tensor (all parameters x dependent parameters)
        Tensor that encodes how complete parameter set which includes dependent
        parameters arises from indepedent parameters: all = <constraints, indep>.
        This form of constraints is used in vector generalized linear models (VGLMs).
    :param design_scale: design matrix for scale
    :param constraints: some design constraints
    :param size_factors: size factors for X
    :param link_fn: linker function for GLM
    :param inv_link_fn: inverse linker function for GLM
    :return: tuple: (groupwise_means, mu, rmsd)
    """
    groupwise_means, m, rmsd1 =  closedform_glm_mean(
        X=X,
        dmat=design_loc,
        constraints=constraints_loc,
        size_factors=size_factors,
        link_fn=link_fn,
        inv_link_fn=inv_link_fn
    )
    mean = np.exp(m)

    groupwise_scale, v, rmsd2 =  closedform_glm_scale(
        X=X,
        design_scale=design_scale,
        constraints=constraints,
        size_factors=size_factors,
        groupwise_means=None,
        link_fn=link_fn,
        compute_scales_fun=None
    )
    var = np.exp(v)
    p = mean / var * (mean * (1-mean) - var)
    print("mean: \n", mean, "\n var: \n", var, "\n p: \n", p)
    return groupwise_means, np.log(p), rmsd1


def closedform_beta_glm_logq(
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
    Calculates a closed-form solution for the `q` parameters of beta GLMs.

    :param X: The sample data
    :param design_loc: design matrix for location
    :param constraints_loc: tensor (all parameters x dependent parameters)
        Tensor that encodes how complete parameter set which includes dependent
        parameters arises from indepedent parameters: all = <constraints, indep>.
        This form of constraints is used in vector generalized linear models (VGLMs).
    :param design_scale: design matrix for scale
    :param constraints: some design constraints
    :param size_factors: size factors for X
    :param link_fn: linker function for GLM
    :param inv_link_fn: inverse linker function for GLM
    :return: tuple: (groupwise_means, mu, rmsd)
    """
    groupwise_means, m, rmsd1 = closedform_glm_mean(
        X=X,
        dmat=design_loc,
        constraints=constraints_loc,
        size_factors=size_factors,
        link_fn=link_fn,
        inv_link_fn=inv_link_fn
    )
    mean = np.exp(m)

    groupwise_scale, v, rmsd2 = closedform_glm_scale(
        X=X,
        design_scale=design_scale,
        constraints=constraints,
        size_factors=size_factors,
        groupwise_means=None,
        link_fn=link_fn,
        compute_scales_fun=None
    )
    var = np.exp(v)

    q = (1 - mean) / var * (mean * (1 - mean) - var)
    return groupwise_scale, np.log(q), rmsd2
