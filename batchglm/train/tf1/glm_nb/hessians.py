import tensorflow as tf

import logging

from .external import HessianGLMALL

logger = logging.getLogger(__name__)


class Hessians(HessianGLMALL):

    def _weight_hessian_ab(
            self,
            X,
            loc,
            scale,
    ):
        if isinstance(X, tf.SparseTensor):
            X_minus_mu = tf.sparse.add(X, -loc)
        else:
            X_minus_mu = X - loc

        const = tf.multiply(
            loc * scale,
            tf.divide(
                X_minus_mu,
                tf.square(loc + scale)
            )
        )

        return const

    def _weight_hessian_aa(
            self,
            X,
            loc,
            scale,
    ):
        if isinstance(X, tf.SparseTensor):
            X_by_scale_plus_one = tf.sparse.add(X.__div__(scale), tf.ones_like(scale))
        else:
            X_by_scale_plus_one = X / scale + tf.ones_like(scale)

        const = tf.negative(tf.multiply(
            loc,
            tf.divide(
                X_by_scale_plus_one,
                tf.square((loc / scale) + tf.ones_like(loc))
            )
        ))

        return const

    def _weight_hessian_bb(
            self,
            X,
            loc,
            scale,
    ):
        if isinstance(X, tf.SparseTensor):
            scale_plus_x = tf.sparse.add(X, scale)
        else:
            scale_plus_x = X + scale

        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        scalar_two = tf.constant(2, shape=(), dtype=self.dtype)
        # Pre-define sub-graphs that are used multiple times:
        scale_plus_loc = scale + loc
        # Define graphs for individual terms of constant term of hessian:
        const1 = tf.add(
            tf.math.digamma(x=scale_plus_x),
            scale * tf.math.polygamma(a=scalar_one, x=scale_plus_x)
        )
        const2 = tf.negative(tf.add(
            tf.math.digamma(x=scale),
            scale * tf.math.polygamma(a=scalar_one, x=scale)
        ))
        const3 = tf.negative(tf.divide(
            tf.add(
                loc * scale_plus_x,
                scalar_two * scale * scale_plus_loc
            ),
            tf.math.square(scale_plus_loc)
        ))
        const4 = tf.add(
            tf.math.log(scale),
            scalar_two - tf.math.log(scale_plus_loc)
        )
        const = tf.add_n([const1, const2, const3, const4])
        const = tf.multiply(scale, const)
        return const


