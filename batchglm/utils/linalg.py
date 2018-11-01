from typing import List

import numpy as np
import xarray as xr

import logging

logger = logging.getLogger(__name__)


def stacked_lstsq(L, b, rcond=1e-10):
    r"""
    Solve `Lx = b`, via SVD least squares cutting of small singular values

    :param L: tensor of shape (..., M, K)
    :param b: tensor of shape (..., M, N).
    :param rcond: threshold for inverse
    :param name: name scope of this op
    :return: x of shape (..., K, N)
    """
    u, s, v = np.linalg.svd(L, full_matrices=False)
    s_max = s.max(axis=-1, keepdims=True)
    s_min = rcond * s_max

    inv_s = np.where(s >= s_min, np.reciprocal(s), 0)

    x = np.einsum(
        '...MK,...MN->...KN',
        v,
        np.einsum('...K,...MK,...MN->...KN', inv_s, u, b)
    )

    # rank = np.sum(s > rcond)

    return np.conj(x, out=x)


def combine_matrices(list_of_matrices: List):
    """
    Combines a list of matrices to a 3D matrix.
    This is done by taking the maximum of all shapes as shape of the 3D matrix and filling all additional values with 0.

    :param list_of_matrices: list of 2D matrices
    :return: matrix of shape (<# matrices>, <max #row in matrices>, <max #cols in matrices>)
    """
    max_num_rows = np.max([mat.shape[0] for mat in list_of_matrices])
    max_num_cols = np.max([mat.shape[1] for mat in list_of_matrices])

    retval = np.zeros([len(list_of_matrices), max_num_rows, max_num_cols])
    for i, mat in enumerate(list_of_matrices):
        retval[i, :mat.shape[0], :mat.shape[1]] = mat

    return retval


def groupwise_solve_lm(
        data: xr.DataArray,
        dmat,
        apply_fun: callable,
        constraints=None,
):
    r"""
    Solve GLMs by estimating the distribution parameters of each unique group of observations independently and
    solving then for the design matrix `dmat`.

    Idea:
    $$
        \theta &= f(x) \\
        \Rightarrow f^{-1}(\theta) &= x \\
            &= (D \cdot D^{+}) \cdot x \\
            &= D \cdot (D^{+} \cdot x) \\
            &= D \cdot x' = f^{-1}(\theta)
    $$

    :param data:
    :param dmat:
    :param apply_fun:
    :param constraints:
    :return:
    """
    unique_design_scale, inverse_idx = np.unique(dmat, axis=0, return_inverse=True)

    if constraints is not None:
        unique_design_constraints = constraints.copy()
        # -1 in the constraint matrix is used to indicate which variable
        # is made dependent so that the constrained is fullfilled.
        # This has to be rewritten here so that the design matrix is full rank
        # which is necessary so that it can be inverted for parameter
        # initialisation.
        unique_design_constraints[unique_design_constraints == -1] = 1
        # Add constraints into design matrix to remove structural unidentifiability.
        unique_design_scale = np.vstack([unique_design_scale, unique_design_constraints])

    if unique_design_scale.shape[1] > np.linalg.matrix_rank(unique_design_scale):
        logger.warning("Scale model is not full rank!")

    grouped_data = data.assign_coords(group=((data.dims[0],), inverse_idx))
    params = apply_fun(grouped_data)

    if constraints is not None:
        param_constraints = np.zeros([constraints.shape[0], params.shape[1]])
        # Add constraints (sum to zero) to value vector to remove structural unidentifiability.
        params = np.vstack([params, param_constraints])

    # inv_design = np.linalg.pinv(unique_design_scale) # NOTE: this is numerically inaccurate!
    # inv_design = np.linalg.inv(unique_design_scale) # NOTE: this is exact if full rank!
    # init_b = np.matmul(inv_design, b)
    #
    # Use least-squares solver to calculate a':
    # This is faster and more accurate than using matrix inversion.
    logger.debug(" ** Solve lstsq problem")
    params_prime = np.linalg.lstsq(unique_design_scale, params, rcond=None)

    return params_prime
