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
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        const = tf.polygamma(scalar_one, loc + scale) * loc * scale

        return const

    def _weight_hessian_aa(
            self,
            X,
            loc,
            scale,
    ):
        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            Xdense = tf.sparse.to_dense(X)
        else:
            Xdense = X

        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        const = loc * (tf.digamma(loc+scale) - tf.digamma(loc) + tf.log(Xdense) +
                       loc*(tf.polygamma(scalar_one, loc+scale) - tf.polygamma(scalar_one, loc)))

        return const

    def _weight_hessian_bb(
            self,
            X,
            loc,
            scale,
    ):
        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            one_minus_X = - tf.sparse.add(X, -tf.ones(shape=X.dense_shape, dtype=self.dtype))
        else:
            one_minus_X = tf.ones_like(X) - X

        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        const = scale * (tf.digamma(loc + scale) - tf.digamma(scale) + tf.log(one_minus_X) +
                         scale * (tf.polygamma(scalar_one, loc+scale) - tf.polygamma(scalar_one, scale)))

        return const

