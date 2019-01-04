import logging

import tensorflow as tf

from .external import JacobiansGLMALL

logger = logging.getLogger(__name__)


class Jacobians(JacobiansGLMALL):
    """
    Compute the analytic Jacobian matrix for a negative binomial GLM.
    """

    def _W_a(
            self,
            X,
            mu,
            r,
    ):
        """
        Compute the coefficient index invariant part of the
        mean model gradient of base_glm_all model.

        Below, X are design matrices of the mean (m)
        and dispersion (r) model respectively, Y are the
        observed data. Const is constant across all combinations
        of i and j.
        .. math::

            &J^{m}_{i} = X^m_i*\bigg(Y-(Y+r)*\frac{mu}{mu+r}\bigg) \\
            &const = Y-(Y+r)*\frac{mu}{mu+r} \\
            &J^{m}_{i} = X^m_i*const \\

        :param X: tf.tensor observations x features
            Observation by observation and feature.
        :param mu: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param r: tf.tensor observations x features
            Value of dispersion model by observation and feature.

        :return const: tf.tensor observations x features
            Coefficient invariant terms of hessian of
            given observations and features.
        """
        const = tf.multiply(  # [observations, features]
            tf.add(X, r),
            tf.divide(
                mu,
                tf.add(mu, r)
            )
        )
        const = tf.subtract(X, const)
        return const


    def _W_b(
            self,
            X,
            mu,
            r,
    ):
        """
        Compute the coefficient index invariant part of the
        dispersion model gradient of base_glm_all model.

        Below, X are design matrices of the mean (m)
        and dispersion (r) model respectively, Y are the
        observed data. Const is constant across all combinations
        of i and j.
        .. math::

            J{r}_{i} &= X^r_i \\
                &*r*\bigg(psi_0(r+Y)-psi_0(r) \\
                &-\frac{r+Y}{r+mu} \\
                &+log(r)+1-log(r+mu) \bigg) \\
            const = r*\bigg(psi_0(r+Y)-psi_0(r) \\ const1
                &-\frac{r+Y}{r+mu} \\ const2
                &+log(r)+1-log(r+mu) \bigg) \\ const3
            J^{r}_{i} &= X^r_i * const \\

        :param X: tf.tensor observations x features
            Observation by observation and feature.
        :param mu: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param r: tf.tensor observations x features
            Value of dispersion model by observation and feature.

        :return const: tf.tensor observations x features
            Coefficient invariant terms of hessian of
            given observations and features.
        """
        scalar_one = tf.constant(1, shape=(), dtype=X.dtype)
        # Pre-define sub-graphs that are used multiple times:
        r_plus_mu = r + mu
        r_plus_x = r + X
        # Define graphs for individual terms of constant term of hessian:
        const1 = tf.subtract(
            tf.math.digamma(x=r_plus_x),
            tf.math.digamma(x=r)
        )
        const2 = tf.negative(r_plus_x / r_plus_mu)
        const3 = tf.add(
            tf.log(r),
            scalar_one - tf.log(r_plus_mu)
        )
        const = tf.add_n([const1, const2, const3])  # [observations, features]
        const = r * const
        return const
