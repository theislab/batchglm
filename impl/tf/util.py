import tensorflow as tf
from typing import Dict, Any, Union, Tuple


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

    retval = None
    if weight is None:
        retval = tf.reduce_mean(input_tensor, axis=axis, name=name, keepdims=True, **kwargs)
    else:
        with tf.name_scope(name):
            retval = tf.reduce_sum(weight * input_tensor,
                                   axis=axis,
                                   keepdims=True,
                                   name="sum_of_fractions",
                                   **kwargs) / tf.reduce_sum(weight,
                                                             axis=axis,
                                                             keepdims=True,
                                                             name="denominator_sum",
                                                             **kwargs)

    if not keepdims:
        retval = tf.squeeze(retval, axis=axis)

    return retval


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


def for_loop(condition, modifier, body_op, idx=0):
    idx = tf.convert_to_tensor(idx)

    def body(i):
        with tf.control_dependencies([body_op(i)]):
            return [modifier(i)]

    # do the loop:
    loop = tf.while_loop(condition, body, [idx])
    return loop


def for_i_in_range(size, body_op):
    loop = for_loop(
        lambda i: tf.less(i, size),
        lambda i: tf.add(i, 1),
        body_op
    )
    return loop


# class IteratorStepper:
#     iterator: tf.data.Iterator
#     variables: Tuple[tf.Variable, ...]
#
#     def __init__(self, iterator: tf.data.Iterator, name="IteratorStepper"):
#         self.iterator = iterator
#
#         iter_values = self.iterator.get_next()
#         if not type(iter_values) == tuple:
#             iter_values = (iter_values,)
#
#         with tf.name_scope(name):
#             with tf.name_scope("variables"):
#                 self.variables = ()
#                 for idx, var in enumerate(iter_values):
#                     self.variables += (
#                         tf.Variable(var, trainable=False, name="var%d" % idx),
#                     )
#
#     def initialized_values(self):
#         retval = ()
#         for idx, var in enumerate(self.variables):
#             retval += (var.initialized_value(),)
#
#         return retval
#
#     def step_op(self):
#         iter_values = self.iterator.get_next()
#         if not type(iter_values) == tuple:
#             iter_values = (iter_values,)
#
#         with tf.name_scope("assign_ops"):
#             assign_ops = ()
#             for idx, (var, value) in enumerate(zip(self.variables, iter_values)):
#                 assign_ops += (
#                     tf.assign(var, value, name="assign%d" % idx),
#                 )
#         return tf.group(assign_ops, name="step_op")


def caching_placeholder(dtype, shape=None, name=None):
    placehldr = tf.placeholder(dtype, shape=shape, name=name)
    var = tf.Variable(placehldr, trainable=False, name=name + "_cache")
    return var


def randomize(tensor, modifier=tf.multiply, min=0.5, max=2.0, name="randomize"):
    with tf.name_scope(name):
        tensor = modifier(tensor, tf.random_uniform(tensor.shape, min, max,
                                                    dtype=tf.float32))
    return tensor
