import tensorflow as tf

import logging

from .external import HessianGLMALL

logger = logging.getLogger(__name__)


class Hessians(HessianGLMALL):
    """
    Compute the analytic model hessian by gene for a negative binomial GLM.
    """

    def _W_ab(
            self,
            X,
            mu,
            r,
    ):
        """
        Compute the coefficient index invariant part of the
        mean-dispersion model block of the hessian of base_glm_all model.

        Note that there are two blocks of the same size which can
        be compute from each other with a transpose operation as
        the hessian is symmetric.

        Below, X are design matrices of the mean (m)
        and dispersion (r) model respectively, Y are the
        observed data. Const is constant across all combinations
        of i and j.
        .. math::

            &H^{m,r}_{i,j} = X^m_i*X^r_j*mu*\frac{Y-mu}{(1+mu/r)^2} \\
            &H^{r,m}_{i,j} = X^m_i*X^r_j*r*mu*\frac{Y-mu}{(mu+r)^2} \\
            &const = r*mu*\frac{Y-mu}{(mu+r)^2} \\
            &H^{m,r}_{i,j} = X^m_i*X^r_j*const \\
            &H^{r,m}_{i,j} = X^m_i*X^r_j*const \\

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
        const = tf.multiply(
            mu * r,  # [observations, features]
            tf.divide(
                X - mu,  # [observations, features]
                tf.square(mu + r)
            )
        )
        return const

    def _W_aa(
            self,
            X,
            mu,
            r,
    ):
        """
        Compute the coefficient index invariant part of the
        mean model block of the hessian of base_glm_all model.

        Below, X are design matrices of the mean (m)
        and dispersion (r) model respectively, Y are the
        observed data. Const is constant across all combinations
        of i and j.
        .. math::

            &H^{m,m}_{i,j} = -X^m_i*X^m_j*mu*\frac{Y/r+1}{(1+mu/r)^2} \\
            &const = -mu*\frac{Y/r+1}{(1+mu/r)^2} \\
            &H^{m,m}_{i,j} = X^m_i*X^m_j*const \\

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
        const = tf.negative(tf.multiply(
            mu,  # [observations, features]
            tf.divide(
                (X / r) + 1,
                tf.square((mu / r) + 1)
            )
        ))
        return const

    def _W_bb(
            self,
            X,
            mu,
            r,
    ):
        """
        Compute the coefficient index invariant part of the
        dispersion model block of the hessian of base_glm_all model.

        Below, X are design matrices of the mean (m)
        and dispersion (r) model respectively, Y are the
        observed data. Const is constant across all combinations
        of i and j.
        .. math::

            H^{r,r}_{i,j}&= X^r_i*X^r_j \\
                &*r*\bigg(psi_0(r+Y)+r*psi_1(r+Y) \\
                &+psi_0(r)+r*psi_1(r) \\
                &-\frac{mu*(r+X)+2*r*(r+m)}{(r+mu)^2} \\
                &+log(r)+1-log(r+mu) \bigg) \\
            const = r*\bigg(psi_0(r+Y)+r*psi_1(r+Y) \\ const1
                &+psi_0(r)+r*psi_1(r) \\ const2
                &-\frac{mu*(r+X)+2*r*(r+m)}{(r+mu)^2} \\ const3
                &+log(r)+1-log(r+mu) \bigg) \\ const4
            H^{r,r}_{i,j}&= X^r_i*X^r_j * const \\

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
        scalar_two = tf.constant(2, shape=(), dtype=X.dtype)
        # Pre-define sub-graphs that are used multiple times:
        r_plus_mu = r + mu
        r_plus_x = r + X
        # Define graphs for individual terms of constant term of hessian:
        const1 = tf.add(  # [observations, features]
            tf.math.digamma(x=r_plus_x),
            r * tf.math.polygamma(a=scalar_one, x=r_plus_x)
        )
        const2 = tf.negative(tf.add(  # [observations, features]
            tf.math.digamma(x=r),
            r * tf.math.polygamma(a=scalar_one, x=r)
        ))
        const3 = tf.negative(tf.divide(
            tf.add(
                mu * r_plus_x,
                scalar_two * r * r_plus_mu
            ),
            tf.square(r_plus_mu)
        ))
        const4 = tf.add(  # [observations, features]
            tf.log(r),
            scalar_two - tf.log(r_plus_mu)
        )
        const = tf.add_n([const1, const2, const3, const4])  # [observations, features]
        const = tf.multiply(r, const)
        return const


