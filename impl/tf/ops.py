import tensorflow as tf


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
    """
    Swaps two dimensions in a given tensor.
    
    :param tensor: The tensor whose axes should be swapped
    :param axis0: The first axis which should be swapped with `axis1`
    :param axis1: The second axis which should be swapped with `axis0`
    :param exec_transpose: Should the transpose operation be applied?
    :param return_perm: Should the permutation argument for `tf.transpose` be returned?
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


def for_loop(condition, modifier, body_fn, idx=0):
    """
    Creates a Tensorflow loop op running `body_fn` in a for loop.
    
    Consider this loop as an equivalent to ```for (idx; condition; modifier): body_fn(idx)```
    
    Example equivalent to ```for (i=0; i < 10; i++): print(i)```:
        for_loop(
            condition = lambda i: tf.less(i, 10),\n
            modifier = lambda i: tf.add(i, 1),\n
            body_fn = lambda i: tf.Print(i, [i]),\n
            idx = 0\n
        )
    
    :param condition: function taking an integer tensor, returning a boolean tensor.
        
        Example: ```lambda i: tf.less(i, 10)```
    :param modifier: function taking an integer tensor, returning another integer tensor.
    
        Example: ```lambda i: tf.add(i, 1)```
    :param body_fn: function taking an integer tensor, returning an Tensorflow operation.
        See tf.while_loop for details.
    :param idx: The initial iterator value / tensor
    :return: tf.while_loop
    """
    idx = tf.convert_to_tensor(idx)
    
    def body(i):
        with tf.control_dependencies([body_fn(i)]):
            return [modifier(i)]
    
    # do the loop:
    loop = tf.while_loop(condition, body, [idx])
    return loop


def for_i_in_range(size, body_fn):
    """
    Creates a Tensorflow loop op running `body_fn` for each value in range(size).
    
    Consider this loop as an equivalent to ```for i in range(size): body_fn(i)```
    
    :param body_fn: function taking an integer tensor, returning an Tensorflow operation.
        See tf.while_loop for details.
    :return: tf.while_loop
    """
    loop = for_loop(
        lambda i: tf.less(i, size),
        lambda i: tf.add(i, 1),
        body_fn
    )
    return loop


def placeholder_variable(dtype, shape=None, name=None):
    """
    Creates a placeholder with name `name` and returns a Variable initialized with this placeholder.
    
    Use this function to remove the necessity to feed data in each session run,
    as it keeps the data in the Variable.
    
    :param dtype: dtype of the placeholder
    :param shape: shape of the placeholder
    :param name:  name of the placeholder
    :return: tf.Variable
    
    """
    placehldr = tf.placeholder(dtype, shape=shape, name=name)
    var = tf.Variable(placehldr, trainable=False, name=name + "_cache")
    return var
