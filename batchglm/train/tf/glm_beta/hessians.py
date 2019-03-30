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
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        const = loc * (tf.digamma(loc+scale) - tf.digamma(loc) + loc*(tf.polygamma(scalar_one, loc+scale) - tf.polygamma(scalar_one, loc)))
        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            const1 = X.__mul__(loc)
            const2 = tf.sparse.add(const1, const)
        else:
            const2 = const + X * loc

        return const2

    def _weight_hessian_bb(
            self,
            X,
            loc,
            scale,
    ):
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        const = scale * (tf.digamma(loc+scale) - tf.digamma(scale) + scale*(tf.polygamma(scalar_one, loc+scale) - tf.polygamma(scalar_one, scale)))
        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            const1 = X.__mul__(scale)
            const2 = tf.sparse.add(const1, const)
        else:
            const2 = const + X * scale

        return const2

