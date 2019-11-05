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
        scalar_two = tf.constant(2, shape=(), dtype=self.dtype)
        if isinstance(X, tf.SparseTensor):
            X_minus_loc = tf.sparse.add(X, -loc)
        else:
            X_minus_loc = X - loc

        const = - tf.multiply(scalar_two,
            tf.divide(
                X_minus_loc,
                tf.square(scale)
            )
        )
        return const

    def _weight_hessian_aa(
            self,
            X,
            loc,
            scale,
    ):
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        const = - tf.divide(scalar_one, tf.square(scale))

        return const

    def _weight_hessian_bb(
            self,
            X,
            loc,
            scale,
    ):
        scalar_two = tf.constant(2, shape=(), dtype=self.dtype)
        if isinstance(X, tf.SparseTensor):
            X_minus_loc = tf.sparse.add(X, -loc)
        else:
            X_minus_loc = X - loc

        const = - tf.multiply(
            scalar_two,
            tf.math.square(
                tf.divide(
                    X_minus_loc,
                    scale
                )
            )
        )
        return const


