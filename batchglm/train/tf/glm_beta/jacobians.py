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
        one_minus_loc = tf.ones_like(loc) - loc
        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            const1 = tf.log(tf.sparse.to_dense(X)/-tf.sparse.add(X, -tf.ones(shape=X.dense_shape, dtype=self.dtype)))
        else:
            const1 = tf.log(X/(tf.ones_like(X)-X))
        const2 = - tf.digamma(loc*scale) + tf.digamma(one_minus_loc*scale) + const1
        const = const2 * scale * loc * one_minus_loc
        return const

    def _weights_jac_b(
            self,
            X,
            loc,
            scale,
    ):
        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            one_minus_X = - tf.sparse.add(X, -tf.ones(shape=X.dense_shape, dtype=self.dtype))
            Xdense = tf.sparse.to_dense(X)
        else:
            one_minus_X = tf.ones_like(X) - X
            Xdense = X
        one_minus_loc = tf.ones_like(X) - loc
        const = scale * (tf.digamma(scale) - tf.digamma(loc*scale)*loc - tf.digamma(one_minus_loc*scale)*one_minus_loc
            + loc * tf.log(Xdense) + one_minus_loc * tf.log(one_minus_X))
        return const
