from typing import List

try:
    import xarray as xr
except ImportError:
    xr = None

import numpy as np


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


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.
    Compatible with xr.DataArrays.

    :param X: input data
    :param theta: float parameter, used as a multiplier prior to exponentiation. Default = 1.0
    :param axis: axis to compute values along. Default is the first non-singleton axis.
    :returns: array of the same size as X. The result will sum to 1 along the specified axis.
    """

    if theta != 1:
        # multiply y against the theta parameter,
        X = X * theta

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(X.shape) if j[1] > 1)

    # take the max along the specified axis
    if isinstance(X, xr.DataArray):
        ax_max = np.max(X, axis=axis)
    else:
        ax_max = np.max(X, axis=axis, keepdims=True)

    # subtract the max for numerical stability
    X = np.exp(X - ax_max)

    # take the sum along the specified axis
    if isinstance(X, xr.DataArray):
        ax_sum = np.sum(X, axis=axis)
    else:
        ax_sum = np.sum(X, axis=axis, keepdims=True)

    # divide elementwise
    p = X / ax_sum

    return p
