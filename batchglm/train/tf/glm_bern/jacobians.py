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
            one_minus_X = - tf.sparse.add(X, -1)
            Xdense = tf.sparse.to_dense(X)
        else:
            one_minus_X = 1 - X
            Xdense = X

        const = Xdense*(1-loc) - (one_minus_X)*loc

        return const

    def _weights_jac_b(
            self,
            X,
            loc,
            scale,
    ):
        return tf.zeros_like(loc)
