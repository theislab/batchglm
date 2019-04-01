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
            Xdense = tf.sparse.to_dense(X)
        else:
            Xdense = X
        const = (tf.digamma(loc+scale) - tf.digamma(loc) + tf.log(Xdense)) * loc

        return const

    def _weights_jac_b(
            self,
            X,
            loc,
            scale,
    ):
        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            one_minus_X = - tf.sparse.add(X, -tf.ones(shape=X.dense_shape, dtype=self.dtype))
        else:
            one_minus_X = tf.ones_like(X) - X


        const = (tf.digamma(loc+scale) - tf.digamma(scale) + tf.log(one_minus_X)) * scale

        return const
