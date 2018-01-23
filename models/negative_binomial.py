import tensorflow as tf
from tensorflow.contrib.distributions import NegativeBinomial

import math

"""
Additional methods related to negative binomial distributions. 
See also :class:`NegativeBinomial`
"""


def fit(sample_data, optimizable=True, name=None):
    """
    Fits negative binomial distributions NB(r, p) to given sample data along axis 0.

    Usage example:
    \`distribution = fit(sample_data);
    :param sample_data: matrix containing samples for each distribution on axis 0\n
        E.g. `(N, M)` matrix with `M` distributions containing `N` observed values each
    :param optimizable: if true, the returned distribution's parameters will be optimizable
    :param name: A name for the operation (optional).
    :return: negative binomial distribution
    """
    with tf.name_scope(name, "fit"):
        (r, p) = fit_mme(sample_data)

        if optimizable:
            r = tf.Variable(r, dtype=tf.float32, validate_shape=False, name="r")
            # r_var = tf.Variable(tf.zeros(tf.shape(r)), dtype=tf.float32, validate_shape=False, name="r_var")
            #
            # r_assign_op = tf.assign(r_var, r)

        # keep mu constant
        mu = tf.reduce_mean(sample_data, axis=0, name="mu")

        # p is directly dependent from mu and r
        p = mu / (r + mu)
        p = tf.identity(p, "p")

        distribution = NegativeBinomial(total_count=r,
                                        probs=p,
                                        name="nb-dist")

        return distribution


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
            replace_values = tf.fill(tf.shape(variance), math.inf, name="inf_constant")

        r_by_mean = tf.where(tf.less(mean, variance),
                             mean / (variance - mean),
                             replace_values)
        r = r_by_mean * mean
        r = tf.identity(r, "r")

        p = 1 / (r_by_mean + 1)
        p = tf.identity(p, "p")

        return r, p
