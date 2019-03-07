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
        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            const1 = tf.sparse.add(X, -loc)
            const2 = tf.divide(-loc, tf.square(scale))
            const = tf.multiply(const1, const2)
        else:
            const1 = tf.subtract(X, loc)
            const2 = tf.divide(-loc, tf.square(scale))
            const = tf.multiply(const1, const2)
        return const

    def _weights_jac_b(
            self,
            X,
            loc,
            scale,
    ):
        # Pre-define sub-graphs that are used multiple times:
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            const = - scalar_one - tf.square(
                tf.divide(tf.sparse.add(X, -loc), scale)
            )
        else:
            const = - scalar_one - tf.square(
                tf.divide(tf.subtract(X, loc), scale)
            )
        return const
