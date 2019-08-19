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
        one_minus_loc = 1 - loc
        if isinstance(X, tf.SparseTensor):
            const1 = tf.log(tf.sparse_add(tf.zeros_like(loc), X).__div__(-tf.sparse.add(X, -tf.ones_like(loc))))
        else:
            const1 = tf.log(X/(1-X))
        const2 = - tf.digamma(loc*scale) + tf.digamma(one_minus_loc*scale) + const1
        const = const2 * scale * loc * one_minus_loc
        return const

    def _weights_jac_b(
            self,
            X,
            loc,
            scale,
    ):
        if isinstance(X, tf.SparseTensor):
            one_minus_X = - tf.sparse.add(X, -tf.ones_like(loc))
        else:
            one_minus_X = 1 - X
        one_minus_loc = 1 - loc
        const = scale * (tf.digamma(scale) - tf.digamma(loc*scale) * loc - tf.digamma(one_minus_loc*scale) * one_minus_loc\
            + loc * tf.log(one_minus_X) + one_minus_loc * tf.log(one_minus_X))
        return const
