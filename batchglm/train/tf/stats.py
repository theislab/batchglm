import tensorflow as tf


def normalize(measure: tf.Tensor, data: tf.Tensor, name="normalize") -> tf.Tensor:
    """
    Normalize measure (e.g. `RMSD` or `MAE`) with the range of `data`

    :param measure: the measure which should be normalized
    :param data: Tensor representing the data by which the measure should be normalized
    :param name: name of this operation
    :return: \frac{RMSD}{max(data) - min(data)}
    """
    with tf.name_scope(name):
        retval = measure / (tf.maximum(data) - tf.minimum(data))
    return retval


def rmsd(estim: tf.Tensor, obs: tf.Tensor, axis=None, name="RMSD") -> tf.Tensor:
    """
    Calculate the root of the mean squared deviation between the estimated and the observed data

    :param estim: Tensor representing the estimated data
    :param obs: Tensor representing the observed data
    :param axis: axis to reduce
    :param name: name of this operation
    :return: \sqrt{mean{(estim - obs)^2}}
    """
    with tf.name_scope(name):
        retval = tf.sqrt(tf.reduce_mean(tf.squared_difference(estim, obs), axis=axis))
    return retval


def mae(estim: tf.Tensor, obs: tf.Tensor, axis=None, name="MAE") -> tf.Tensor:
    """
    Calculate the mean absolute error between the estimated weights `b` and the true `b`

    :param estim: Tensor representing the estimated data
    :param obs: Tensor representing the observed data
    :param axis: axis to reduce
    :param name: name of this operation
    :return: mean{|estim - obs|}
    """
    with tf.name_scope(name):
        retval = tf.reduce_mean(tf.abs(estim - obs), axis=axis)
    return retval


def normalized_rmsd(estim: tf.Tensor, obs: tf.Tensor, axis=None, name="NRMSD") -> tf.Tensor:
    """
    Calculate the normalized RMSD between estimated and observed data

    :param estim: Tensor representing the estimated data
    :param obs: Tensor representing the observed data
    :param axis: axis to reduce
    :param name: name of this operation
    :return: \frac{RMSD}{max(obs) - min(obs)}
    """
    with tf.name_scope(name):
        retval = normalize(rmsd(estim, obs, axis=axis), obs)
    return retval


def normalized_mae(estim: tf.Tensor, obs: tf.Tensor, axis=None, name="NMAE") -> tf.Tensor:
    """
    Calculate the normalized MAE between estimated and observed data

    :param estim: Tensor representing the estimated data
    :param obs: Tensor representing the observed data
    :param axis: axis to reduce
    :param name: name of this operation
    :return: \frac{MAE}{max(obs) - min(obs)}
    """
    with tf.name_scope(name):
        retval = normalize(mae(estim, obs, axis=axis), obs)
    return retval


def mapd(estim: tf.Tensor, obs: tf.Tensor, axis=None, name="MAPD") -> tf.Tensor:
    """
    Calculate the mean absolute percentage deviation between the estimated and the observed data

    :param estim: ndarray representing the estimated data
    :param obs: ndarray representing the observed data
    :param axis: axis to reduce
    :param name: name of this operation
    :return: mean{|estim - obs| / obs}
    """
    with tf.name_scope(name):
        retval = tf.reduce_mean(tf.abs(estim - obs) / obs, axis=axis)
    return retval
