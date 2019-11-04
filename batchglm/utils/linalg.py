import dask.array
import numpy as np

import logging

logger = logging.getLogger("batchglm")


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
        constraints: np.ndarray
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

    :param dmat: design matrix which should be solved for
    :param apply_fun: some callable function taking one xr.DataArray argument.
        Should compute a group-wise parameter solution.

        Example method calculating group-wise means:
        ::
            def apply_fun(grouping):
                groupwise_means = data.groupby(grouping).mean(dim="observations").values

                return np.log(groupwise_means)

        The `data` argument provided to `apply_fun` is the same xr.DataArray provided to this
    :param constraints: tensor (all parameters x dependent parameters)
        Tensor that encodes how complete parameter set which includes dependent
        parameters arises from indepedent parameters: all = <constraints, indep>.
        This form of constraints is used in vector generalized linear models (VGLMs).

    :return: tuple of (apply_fun(grouping), x_prime, rmsd, rank, s) where x_prime is the parameter matrix solved for
    `dmat`.
    """
    # Get unqiue rows of design matrix and vector with group assignments:
    if isinstance(dmat, dask.array.core.Array):  # axis argument not supported by dask in .unique()
        unique_design, inverse_idx = np.unique(dmat.compute(), axis=0, return_inverse=True)
        unique_design = dask.array.from_array(unique_design, chunks=unique_design.shape)
    else:
        unique_design, inverse_idx = np.unique(dmat, axis=0, return_inverse=True)
    if unique_design.shape[0] > 100:
        raise ValueError("large least-square problem in init, likely defined a numeric predictor as categorical")

    full_rank = constraints.shape[1]
    if isinstance(dmat, dask.array.core.Array):  # matrix_rank not supported by dask
        rank = np.linalg.matrix_rank(np.matmul(unique_design.compute(), constraints.compute()))
    else:
        rank = np.linalg.matrix_rank(np.matmul(unique_design, constraints))
    if full_rank > rank:
        logger.error("model is not full rank!")

    # Get group-wise means in linker space based on group assignments
    # based on unique rows of design matrix:
    params = apply_fun(inverse_idx)

    # Use least-squares solver to compute model parameterization
    # accounting for dependent parameters, ie. degrees of freedom
    # of the model which appear as groups in the design matrix
    # and are not accounted for by parameters but which are
    # accounted for by constraints:
    # <X, <theta, H> = means -> <X, theta>, H> = means -> lstsqs for theta
    # (This is faster and more accurate than using matrix inversion.)
    logger.debug(" ** Solve lstsq problem")
    if np.any(np.isnan(params)):
        raise Warning("entries of params were nan which will throw error in lstsq")
    x_prime, rmsd, rank, s = np.linalg.lstsq(
        np.matmul(unique_design, constraints),
        params
    )

    return params, x_prime, rmsd, rank, s


