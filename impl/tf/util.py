import tensorflow as tf
from typing import Dict, Any, Union


def input_to_feed_dict(graph, input_data: dict) -> Dict[Union[Union[tf.Tensor, tf.Operation], Any], Any]:
    retval = {}
    with graph.as_default():
        for (key, value) in input_data.items():
            retval[graph.get_tensor_by_name(key + ":0")] = value

    return retval


def reduce_weighted_mean(input_tensor, weight=None, axis=None, keepdims=False, name="mean", **kwargs):
    """
    Calculates the weighted mean of `input_tensor`. See also tf.reduce_mean

    :param input_tensor: tensor to be reduced
    :param weight: the weights of the tensors' elements; if `none` it will be ignored
    :param axis: The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)].
    :param keepdims: If true, retains reduced dimensions with length 1
    :param kwargs: further arguments which will be passed to  `tf.reduce_mean`
    :return: tensor with the weighted absolute mean

    .. seealso:: :py:meth:`reduce_mean()` in module :py:mod:`tensorflow`
    """

    retVal = None
    if weight is None:
        retVal = tf.reduce_mean(input_tensor, axis=axis, name=name, keepdims=True, **kwargs)
    else:
        with tf.name_scope(name):
            retVal = tf.reduce_sum(weight * input_tensor,
                                   axis=axis,
                                   keepdims=True,
                                   name="sum_of_fractions",
                                   **kwargs) / tf.reduce_sum(weight,
                                                             axis=axis,
                                                             keepdims=True,
                                                             name="denominator_sum",
                                                             **kwargs)

    if not keepdims:
        retVal = tf.squeeze(retVal, axis=axis)

    return retVal


def logit(input_tensor, name="logit"):
    with tf.name_scope(name):
        return tf.log(input_tensor / (1 - input_tensor))


def swap_dims(tensor, axis0, axis1, exec_transpose=True, return_perm=False, name="swap_dims"):
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
