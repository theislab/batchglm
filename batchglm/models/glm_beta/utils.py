import numpy as np
import scipy.sparse
from typing import Union

from .external import closedform_glm_mean, closedform_glm_scale


def closedform_beta_glm_logitmean(
        x: Union[np.ndarray, scipy.sparse.csr_matrix],
        design_loc: np.ndarray,
        constraints_loc,
        size_factors=None,
        link_fn=lambda x: np.log(1/(1/x-1)),
        inv_link_fn=lambda x: 1/(1+np.exp(-x))
):
    r"""
    Calculates a closed-form solution for the `mean` parameters of beta GLMs.

    :param x: The sample data
    :param design_loc: design matrix for location
    :param constraints: tensor (all parameters x dependent parameters)
        Tensor that encodes how complete parameter set which includes dependent
        parameters arises from indepedent parameters: all = <constraints, indep>.
        This form of constraints is used in vector generalized linear models (VGLMs).
    :param size_factors: size factors for X
    :return: tuple: (groupwise_means, mean, rmsd)
    """
    return closedform_glm_mean(
        x=x,
        dmat=design_loc,
        constraints=constraints_loc,
        size_factors=size_factors,
        link_fn=link_fn,
        inv_link_fn=inv_link_fn
    )


def closedform_beta_glm_logsamplesize(
        x: Union[np.ndarray, scipy.sparse.csr_matrix],
        design_scale: np.ndarray,
        constraints=None,
        size_factors=None,
        groupwise_means=None,
        link_fn=np.log,
        invlink_fn=np.exp
):
    r"""
    Calculates a closed-form solution for the log-scale parameters of beta GLMs.

    :param x: The sample data
    :param design_scale: design matrix for scale
    :param constraints: some design constraints
    :param size_factors: size factors for X
    :param groupwise_means: optional, in case if already computed this can be specified to spare double-calculation
    :return: tuple (groupwise_scales, logsd, rmsd)
    """

    def compute_scales_fun(variance, mean):
        groupwise_scales = mean*(1-mean)/variance - 1
        return groupwise_scales

    return closedform_glm_scale(
        x=x,
        design_scale=design_scale,
        constraints=constraints,
        size_factors=size_factors,
        groupwise_means=groupwise_means,
        link_fn=link_fn,
        inv_link_fn=invlink_fn,
        compute_scales_fun=compute_scales_fun
    )
