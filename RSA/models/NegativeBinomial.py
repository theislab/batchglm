import tensorflow as tf
import tensorflow.contrib as tfcontrib

import math

"""
Additional methods related to negative binomial distributions. 
See also :class:`tf.contrib.distributions.NegativeBinomial`
"""


def fit(sample_data, optimizable=True, name=None):
    """
    Fits negative binomial distributions NB(r, p) to given sample data along axis 0

    :param sample_data: matrix containing samples for each distribution on axis 0\n
        E.g. `(N, M)` matrix with `M` distributions containing `N` observed values each
    :param optimizable: if true, the returned parameters will be optimizable
    :param name: A name for the operation (optional).
    :return: shape `r`, probability for success `p` and observed mean `mu` for sample data.\n
        `r` and `p` will be initialized with the Maximum-of-Momentum estimator
    """
    with tf.name_scope(name, "fit"):
        (r, p) = fit_mme(sample_data)

        if optimizable:
            r = tf.Variable(r, dtype=tf.float32, name="r")

        # keep mu constant
        mu = tf.reduce_mean(sample_data, axis=0, name="mu")

        # p is directly dependent from mu and r
        p = mu / (r + mu)
        p = tf.identity(p, "p")

        return r, p, mu


def fit_mme(sample_data, replace_values=None, name=None):
    """
        Calculates the Maximum-of-Momentum Estimator of `NB(r, p)` for given sample data along axis 0.

        :param sample_data: matrix containing samples for each distribution on axis 0\n
            E.g. `(N, M)` matrix with `M` distributions containing `N` observed values each
        :param replace_values: Matrix of size `shape(sample_data)[1:]`
        :param name: A name for the operation (optional).
        :return: estimated values of `r` and `p`
        """
    with tf.name_scope(name, "MME"):
        mean = tf.reduce_mean(sample_data, axis=0, name="mean")
        variance = tf.reduce_mean(tf.square(sample_data - mean),
                                  axis=0,
                                  name="variance")
        if replace_values is None:
            replace_values = tf.fill(tf.shape(variance), math.nan, name="NaN_constant")

        r_by_mean = tf.where(tf.less(mean, variance),
                             mean / (variance - mean),
                             replace_values)
        r = r_by_mean * mean
        r = tf.identity(r, "r")

        p = 1 / (r_by_mean + 1)
        p = tf.identity(p, "p")

        return r, p
