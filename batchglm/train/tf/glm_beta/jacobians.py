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
            const = tf.sparse.add(X, tf.digamma(loc+scale) - tf.digamma(loc))
        else:
            const = tf.digamma(loc+scale) - tf.digamma(loc) + X
        const1 = const * loc

        return const1

    def _weights_jac_b(
            self,
            X,
            loc,
            scale,
    ):
        # Pre-define sub-graphs that are used multiple times:
        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            const = - tf.sparse_add(X, - tf.digamma(loc+scale) + tf.digamma(scale) -tf.ones(shape=X.dense_shape, dtype=self.dtype))
        else:
            const = tf.digamma(loc+scale) - tf.digamma(scale) + tf.ones_like(X) - X
        const1 = const * scale

        return const1
