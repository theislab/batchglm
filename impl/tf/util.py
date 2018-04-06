import tensorflow as tf


def reduce_weighted_mean(input_tensor, weight=None, **kwargs):
    """
    Calculates the weighted mean of `input_tensor`. See also tf.reduce_mean
    
    :param input_tensor: tensor to be reduced
    :param weight: the weights of the tensors' elements; if `none` it will be ignored
    :param kwargs: further arguments which will be passed to  `tf.reduce_mean`
    :return: tensor with the weighted absolute mean
    
    .. seealso:: :py:meth:`reduce_mean()` in module :py:mod:`tensorflow`
    """
    if weight is None:
        return tf.reduce_mean(input_tensor, **kwargs)
    else:
        return tf.reduce_mean(weight * input_tensor, **kwargs) / tf.reduce_sum(weight, **kwargs)
