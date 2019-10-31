import tensorflow as tf
from typing import Union


def swap_dims(tensor, axis0, axis1, exec_transpose=True, return_perm=False, name="swap_dims"):
    """
    Swaps two dimensions in a given tensor.

    :param tensor: The tensor whose axes should be swapped
    :param axis0: The first axis which should be swapped with `axis1`
    :param axis1: The second axis which should be swapped with `axis0`
    :param exec_transpose: Should the transpose operation be applied?
    :param return_perm: Should the permutation argument for `tf1.transpose` be returned?
        Autmoatically true, if `exec_transpose` is False
    :param name: The name scope of this op
    :return: either retval, (retval, permutation) or permutation
    """
    with tf.name_scope(name):
        rank = tf.range(tf.rank(tensor))
        idx0 = rank[axis0]
        idx1 = rank[axis1]
        perm0 = tf.where(tf.equal(rank, idx0), tf.tile(tf.expand_dims(idx1, 0), [tf.size(rank)]), rank)
        perm1 = tf.where(tf.equal(rank, idx1), tf.tile(tf.expand_dims(idx0, 0), [tf.size(rank)]), perm0)

    if exec_transpose:
        retval = tf.transpose(tensor, perm1)

        if return_perm:
            return retval, perm1
        else:
            return retval
    else:
        return perm1


def stacked_lstsq(L, b, rcond=1e-10, name="stacked_lstsq"):
    r"""
    Solve `Lx = b`, via SVD least squares cutting of small singular values

    :param L: tensor of shape (..., M, K)
    :param b: tensor of shape (..., M, N).
    :param rcond: threshold for inverse
    :param name: name scope of this op
    :return: x of shape (..., K, N)
    """
    with tf.name_scope(name):
        u, s, v = tf.linalg.svd(L, full_matrices=False)
        s_max = s.max(axis=-1, keepdims=True)
        s_min = rcond * s_max

        inv_s = tf.where(s >= s_min, tf.reciprocal(s), 0)

        x = tf.einsum(
            '...MK,...MN->...KN',
            v,
            tf.einsum('...K,...MK,...MN->...KN', inv_s, u, b)
        )

        return tf.conj(x)
