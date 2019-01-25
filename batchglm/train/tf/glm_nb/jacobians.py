import logging

import tensorflow as tf

from .external import JacobiansGLMALL

logger = logging.getLogger(__name__)


class Jacobians(JacobiansGLMALL):

    def _W_a(
            self,
            X,
            mu,
            r,
    ):
        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            const = tf.multiply(
                tf.sparse.add(X, r),
                tf.divide(
                    mu,
                    tf.add(mu, r)
                )
            )
            const = tf.sparse.add(X, -const)
        else:
            const = tf.multiply(
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
        # Pre-define sub-graphs that are used multiple times:
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            r_plus_x = tf.sparse.add(X, r)
        else:
            r_plus_x = r + X

        r_plus_mu = r + mu

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
