from typing import Union

import tensorflow as tf


def reduce_sum(input_tensor, axis=None, keepdims=False, name="sum") -> Union[tf.Tensor, tf.SparseTensor]:
    if isinstance(input_tensor, tf.SparseTensor):
        with tf.name_scope(name):
            return tf.sparse_reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)
    else:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims, name=name)


def reduce_mean(input_tensor, axis=None, keepdims=False, name="mean") -> Union[tf.Tensor, tf.SparseTensor]:
    if isinstance(input_tensor, tf.SparseTensor):
        with tf.name_scope(name):
            size = tf.size(input_tensor) if axis is None else tf.cumprod(tf.gather(tf.shape(input_tensor), axis))
            sum = reduce_sum(input_tensor, axis=None, keepdims=False)
            return sum / tf.cast(size, dtype=sum.dtype)
    else:
        return tf.reduce_mean(input_tensor, axis=axis, keepdims=keepdims, name=name)


def reduce_weighted_mean(input_tensor, weight=None, axis=None, keepdims=False, name="weighted_mean"):
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
        retval = reduce_mean(input_tensor, axis=axis, name=name, keepdims=True)
    else:
        with tf.name_scope(name):
            retval = reduce_sum(weight * input_tensor,
                                axis=axis,
                                keepdims=True,
                                name="sum_of_fractions") / reduce_sum(weight,
                                                                      axis=axis,
                                                                      keepdims=True,
                                                                      name="denominator_sum")

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


def map_reduce(last_elem: tf.Tensor, data: tf.data.Dataset, map_fn, reduce_fn=tf.add, **kwargs):
    """
    Iterate over elements in a tf.data.Dataset.
    Fetches new elements until "last_elem" appears at `idx[-1]`.

    TODO: remove 'last_elem' as soon as tensorflow iterators support some `has_next` functionality

    :param last_elem: the last element
    :param data: tf.data.Dataset containing `(idx, val)` with idx as a vector of shape `(batch_size,)`
    :param map_fn: function taking arguments `(idx, val)`
    :param reduce_fn: function taking two return values of `map_fn` and reducing them into one return value
    :param kwargs: additional arguments passed to the `tf.while loop`
    :return:
    """
    iterator = data.make_initializable_iterator()

    def cond(idx, val):
        return tf.not_equal(tf.gather(idx, tf.size(idx) - 1), last_elem)

    def body_fn(old_idx, old_val):
        idx, val = iterator.get_next()

        return idx, reduce_fn(old_val, map_fn(idx, val))

    def init_vals():
        idx, val = iterator.get_next()
        return idx, map_fn(idx, val)

    with tf.control_dependencies([iterator.initializer]):
        _, reduced = tf.while_loop(cond, body_fn, init_vals(), **kwargs)

    return reduced


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


def randomize(tensor, modifier=tf.multiply, min=0.5, max=2.0, name="randomize"):
    """
    Randomizes a tensor by applying some random uniform bias

    :param tensor: The tensor which should be randomized
    :param modifier: The modifier function, should take two tensors as arguments. `tf.multiply` per default.
    :param min: minimum bias value
    :param max: maximum bias value
    :param name: name of this operation
    :return: The randomized tensor returned by `modifier`
    """
    with tf.name_scope(name):
        tensor = modifier(tensor, tf.random_uniform(tensor.shape, min, max,
                                                    dtype=tf.float32))
    return tensor


def keep_const(tensor: tf.Tensor, cond: tf.Tensor, constant: tf.Tensor = None, name="keep_const"):
    """
    Keeps some parts of a tensor constant during training ops by replacing the parts defined by `cond`

    :param tensor: The tensor which should be partially kept constant
    :param cond: The condition, e.g. a boolean tensor (See tf.where)
    :param constant: replacement value(s).

        If None, `tf.stop_gradient(tensor)` will be used instead
    :param name: The name scope of this operation
    :return: tf.Tensor with non-trainable parts
    """
    with tf.name_scope(name):
        if constant is None:
            constant = tf.stop_gradient(tensor)

        constant = tf.broadcast_to(constant, shape=cond)
        return tf.where(cond, tensor, constant)


def caching_placeholder(dtype, shape=None, name=None):
    """
    Placeholder which keeps its data after initialization.
    Saves feeding the data in each session run.

    :param dtype: data type of the placeholder
    :param shape: shape of the placeholder
    :param name: name of the placeholder
    :return: tf.Variable, initialized by the placeholder
    """
    placehldr = tf.placeholder(dtype, shape=shape, name=name)

    var = tf.Variable(placehldr, trainable=False, name=name + "_cache")
    return var


def hessian_diagonal(ys, xs, name="hessian_diagonal", **kwargs):
    """
    Returns the second order derivative of ys wrt. xs.
    See tf.gradients() for more details.

    :param ys: values
    :param xs: (list of) variables
    :param name: name of this operation
    :param kwargs: further arguments which will be passed to tf.gradient()
    :return: (list of) tensor(s) corresponding to the variable(s) passed in `xs`
    """
    with tf.name_scope(name):
        return tf.gradients(tf.gradients(ys, xs, name="first_order", **kwargs), xs, name="second_order", **kwargs)


def pinv(matrix, threshold=1e-5):
    """
    Calculate the Moore-Penrose pseudo-inverse of the last two dimensions of some matrix.

    E.g. if `matrix` has some shape [..., K, L, M, N], this method will inverse each [M, N] matrix.

    :param matrix: The matrix to invert
    :param threshold: threshold value
    :return: the pseudo-inverse of `matrix`
    """

    s, u, v = tf.svd(matrix)  # , full_matrices=True, compute_uv=True)

    adj_threshold = tf.reduce_max(s, axis=-1, keepdims=True) * threshold
    s_inv = tf.where(s > tf.broadcast_to(adj_threshold, s.shape), tf.reciprocal(s), tf.zeros_like(s))
    s_inv = tf.matrix_diag(s_inv)

    return v @ (s_inv @ swap_dims(u, axis0=-1, axis1=-2))


def jacobian(fx, x, **kwargs):
    """
    Given a tensor fx, which is a function of x, vectorize fx (via tf.reshape(fx, [-1])),
    and then compute the jacobian of each entry of fx with respect to x.
    Specifically, if x has shape (m,n,...,p), and fx has L entries (tf.size(fx)=L), then
    the output will be (L,m,n,...,p), where output[i] will be (m,n,...,p), with each entry denoting the
    gradient of output[i] wrt the corresponding element of x.
    """
    return tf.map_fn(
        fn=lambda fxi: tf.gradients(fxi, x)[0],
        elems=tf.reshape(fx, [-1]),
        dtype=x.dtype,
        **kwargs
    )
