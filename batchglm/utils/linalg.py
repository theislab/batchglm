from typing import List

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
