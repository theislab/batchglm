import logging

import tensorflow as tf

from .external import JacobiansGLMALL

logger = logging.getLogger(__name__)


class Jacobians(JacobiansGLMALL):

    def _weights_jac_a(
            self,
            X,
            loc,
            scale,
    ):
        if isinstance(X, tf.SparseTensor):
            const1 = tf.sparse.add(X, -loc)
            const = tf.divide(const1, tf.square(scale))
        else:
            const1 = tf.subtract(X, loc)
            const = tf.divide(const1, tf.square(scale))
        return const

    def _weights_jac_b(
            self,
            X,
            loc,
            scale,
    ):
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        if isinstance(X, tf.SparseTensor):
            const = tf.negative(scalar_one) + tf.math.square(
                tf.divide(tf.sparse.add(X, -loc), scale)
            )
        else:
            const = tf.negative(scalar_one) + tf.math.square(
                tf.divide(tf.subtract(X, loc), scale)
            )
        return const
