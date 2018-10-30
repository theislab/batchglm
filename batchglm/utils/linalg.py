import numpy as np


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

    return np.conj(x, out=x)
