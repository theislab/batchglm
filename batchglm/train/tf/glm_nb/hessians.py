import tensorflow as tf

import logging

from .external import HessianGLMALL

logger = logging.getLogger(__name__)


class Hessians(HessianGLMALL):

    def _W_ab(
            self,
            X,
            mu,
            r,
    ):
        const = tf.multiply(
            mu * r,
            tf.divide(
                X - mu,
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
        const = tf.negative(tf.multiply(
            mu,
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
        scalar_one = tf.constant(1, shape=(), dtype=X.dtype)
        scalar_two = tf.constant(2, shape=(), dtype=X.dtype)
        # Pre-define sub-graphs that are used multiple times:
        r_plus_mu = r + mu
        r_plus_x = r + X
        # Define graphs for individual terms of constant term of hessian:
        const1 = tf.add(
            tf.math.digamma(x=r_plus_x),
            r * tf.math.polygamma(a=scalar_one, x=r_plus_x)
        )
        const2 = tf.negative(tf.add(
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
        const4 = tf.add(
            tf.log(r),
            scalar_two - tf.log(r_plus_mu)
        )
        const = tf.add_n([const1, const2, const3, const4])
        const = tf.multiply(r, const)
        return const


