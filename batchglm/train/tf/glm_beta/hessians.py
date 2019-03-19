import tensorflow as tf

import logging

from .external import HessianGLMALL

logger = logging.getLogger(__name__)


class Hessians(HessianGLMALL):

    def _weight_hessian_aa(
            self,
            X,
            loc,
            scale,
    ):
        one_minus_loc = 1 - loc
        loc_times_scale = loc * scale
        one_minus_loc_times_scale = one_minus_loc * scale
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)

        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            const1 = tf.log(tf.sparse.to_dense(X) / -tf.sparse.add(X, -1))
        else:
            const1 = tf.log(X / (1 - X))

        const2 = (1 - 2 * loc) * (- tf.digamma(loc_times_scale) + tf.digamma(one_minus_loc_times_scale) + const1)
        const3 = loc * one_minus_loc_times_scale * (- tf.polygamma(scalar_one, loc_times_scale) - tf.polygamma(scalar_one, one_minus_loc_times_scale))
        const = loc * one_minus_loc_times_scale * (const2 + const3)
        return const

    def _weight_hessian_ab(
            self,
            X,
            loc,
            scale,
    ):
        one_minus_loc = 1 - loc
        loc_times_scale = loc * scale
        one_minus_loc_times_scale = one_minus_loc * scale
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)

        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            const1 = tf.log(tf.sparse.to_dense(X) / -tf.sparse.add(X, -1))
        else:
            const1 = tf.log(X / (1 - X))

        const2 = - tf.digamma(loc_times_scale) + tf.digamma(one_minus_loc_times_scale) + const1
        const3 = scale * (- tf.polygamma(scalar_one, loc_times_scale) * loc + one_minus_loc * tf.polygamma(scalar_one, one_minus_loc_times_scale))

        const = loc * one_minus_loc_times_scale * (const2 + const3)

        return const

    def _weight_hessian_bb(
            self,
            X,
            loc,
            scale,
    ):
        one_minus_loc = 1 - loc
        loc_times_scale = loc * scale
        one_minus_loc_times_scale = one_minus_loc * scale
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)

        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            const1 = tf.log(tf.sparse.to_dense(X) / -tf.sparse.add(X, -1))
        else:
            const1 = tf.log(X / (1 - X))

        const2 = loc * (tf.log(X) - tf.digamma(loc_times_scale))\
                 - one_minus_loc * (tf.digamma(one_minus_loc_times_scale) + tf.log(const1)) \
                 + tf.digamma(scale)
        const3 = scale * (- tf.square(loc) * tf.polygamma(scalar_one, loc_times_scale)\
                          + tf.polygamma(scalar_one, scale)\
                          - tf.polygamma(scalar_one, one_minus_loc_times_scale) * tf.square(one_minus_loc))
        const = scale * (const2 + const3)

        return const


