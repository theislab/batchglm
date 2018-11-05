import numpy as np

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

    inv_s = np.reciprocal(s, out=np.zeros_like(s), where=s >= s_min)

    x = np.einsum(
        '...MK,...MN->...KN',
        v,
        np.einsum('...K,...MK,...MN->...KN', inv_s, u, b)
    )

    # rank = np.sum(s > rcond)

    return np.conj(x, out=x)


def groupwise_solve_lm(
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

    :param data: xr.DataArray of shape (observations, features) which can be grouped by `dmat`
    :param dmat: design matrix which should be solved for
    :param apply_fun: some callable function taking one xr.DataArray argument.
        Should compute a group-wise parameter solution.

        Example method calculating group-wise means:
        ::
            def apply_fun(grouping):
                groupwise_means = data.groupby(grouping).mean(dim="observations").values

                return np.log(groupwise_means)

        The `data` argument provided to `apply_fun` is the same xr.DataArray provided to this

    :param constraints: possible design constraints for constraint optimization
    :return: tuple of (apply_fun(grouping), x_prime, rmsd, rank, s) where x_prime is the parameter matrix solved for
    `dmat`.
    """
    unique_design, inverse_idx = np.unique(dmat, axis=0, return_inverse=True)

    if constraints is not None:
        design_constraints = constraints.copy()
        # -1 in the constraint matrix is used to indicate which variable
        # is made dependent so that the constrained is fullfilled.
        # This has to be rewritten here so that the design matrix is full rank
        # which is necessary so that it can be inverted for parameter
        # initialisation.
        design_constraints[design_constraints == -1] = 1
        # Add constraints into design matrix to remove structural unidentifiability.
        unique_design = np.vstack([unique_design, design_constraints])

    if unique_design.shape[1] > np.linalg.matrix_rank(unique_design):
        logger.warning("model is not full rank!")

    params = apply_fun(inverse_idx)

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
    x_prime, rmsd, rank, s = np.linalg.lstsq(unique_design, params, rcond=None)

    return params, x_prime, rmsd, rank, s


