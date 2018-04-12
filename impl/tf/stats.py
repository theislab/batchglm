import tensorflow as tf


def normalize(measure: tf.Tensor, data: tf.Tensor, name="normalize") -> tf.Tensor:
    """
    Normalize measure (e.g. `RMSD` or `MAE`) with the range of `data`

    :param measure: the measure which should be normalized
    :param data: Tensor representing the data by which the measure should be normalized
    :return: \frac{RMSD}{max(data) - min(data)}
    """
    with tf.name_scope(name):
        norm = measure / (tf.maximum(data) - tf.minimum(data))
    return norm


def rmsd(estim: tf.Tensor, obs: tf.Tensor, name="RMSD") -> tf.Tensor:
    """
    Calculate the root of the mean squared deviation between the estimated and the observed data

    :param estim: Tensor representing the estimated data
    :param obs: Tensor representing the observed data
    :return: \sqrt{mean{(estim - obs)^2}}
    """
    with tf.name_scope(name):
        rmsd = tf.sqrt(tf.reduce_mean(tf.squared_difference(estim, obs)))
    return rmsd


def mae(estim: tf.Tensor, true_b: tf.Tensor, name="MAE") -> tf.Tensor:
    """
    Calculate the mean absolute error between the estimated weights `b` and the true `b`

    :param estim: Tensor representing the estimated data
    :param obs: Tensor representing the observed data
    :return: mean{(estim - obs)}
    """
    with tf.name_scope(name):
        mae = tf.reduce_mean(tf.abs(estim - true_b))
    return mae


def normalized_rmsd(estim: tf.Tensor, obs: tf.Tensor, name="NRMSD") -> tf.Tensor:
    """
    Calculate the normalized RMSD between estimated and observed data

    :param estim: Tensor representing the estimated data
    :param obs: Tensor representing the observed data
    :return: \frac{RMSD}{max(obs) - min(obs)}
    """
    with tf.name_scope(name):
        retval = normalize(rmsd(estim, obs), obs)
    return retval


def normalized_mae(estim: tf.Tensor, obs: tf.Tensor, name="NMAE") -> tf.Tensor:
    """
    Calculate the normalized MAE between estimated and observed data

    :param estim: Tensor representing the estimated data
    :param obs: Tensor representing the observed data
    :return: \frac{MAE}{max(obs) - min(obs)}
    """
    with tf.name_scope(name):
        retval = normalize(mae(estim, obs), obs)
    return retval
