import tensorflow as tf

import logging

from .external import FIMGLMALL

logger = logging.getLogger(__name__)


class FIM(FIMGLMALL):
    """
    Compute expected fisher information matrix (FIM)
    for iteratively re-weighted least squares (IWLS or IRLS) parameter updates
    for a negative binomial GLM.
    """

    def _W_aa(
            self,
            mu,
            r
    ):
        """
        Compute for mean model IWLS update for a negative binomial GLM.

        :param X: tf.tensor observations x features
            Observation by observation and feature.
        :param mu: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param r: tf.tensor observations x features
            Value of dispersion model by observation and feature.

        :return tuple of tf.tensors
            Constants with respect to coefficient index for
            Fisher information matrix and score function computation.
        """
        const = tf.divide(r, r+mu)
        W = tf.multiply(mu, const)

        return W

    def _W_bb(
            self,
            X,
            mu,
            r
    ):
        """
        Compute for dispersion model IWLS update for a negative binomial GLM.

        :param X: tf.tensor observations x features
            Observation by observation and feature.
        :param mu: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param r: tf.tensor observations x features
            Value of dispersion model by observation and feature.
        :param log_rr: tf.tensor observations x features
            Logarithm of dispersion model by observation and feature.

        :return tuple of tf.tensors
            Constants with respect to coefficient index for
            Fisher information matrix and score function computation.
        """
        scalar_one = tf.constant(1, shape=(), dtype=X.dtype)
        scalar_two = tf.constant(2, shape=(), dtype=X.dtype)

        r_plus_mu = r+mu
        digamma_r = tf.math.digamma(x=r)
        digamma_r_plus_mu = tf.math.digamma(x=r_plus_mu)

        const1 = tf.multiply(scalar_two, tf.add(  # [observations, features]
            digamma_r,
            digamma_r_plus_mu
        ))
        const2 = tf.multiply(r, tf.add(  # [observations, features]
            tf.math.polygamma(a=scalar_one, x=r),
            tf.math.polygamma(a=scalar_one, x=r_plus_mu)
        ))
        const3 = tf.divide(r, r_plus_mu)
        W = tf.multiply(r, tf.add_n([const1, const2, const3]))

        return W
