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
        scalar_two = tf.constant(2, shape=(), dtype=self.dtype)
        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            X_minus_loc = tf.sparse.add(X, -loc)
        else:
            X_minus_loc = X - loc

        const = - tf.multiply(scalar_two * loc,
            tf.divide(
                X_minus_loc,
                tf.pow(scale, 3)
            )
        )
        return const

    def _weight_hessian_aa(
            self,
            X,
            loc,
            scale,
    ):
        scalar_two = tf.constant(2, shape=(), dtype=self.dtype)
        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            const = tf.multiply(
                loc / tf.square(scale),
                tf.sparse.add(X, -scalar_two * loc)
            )
        else:
            const = tf.multiply(
                loc / tf.square(scale),
                X - scalar_two * loc
            )

        return const

    def _weight_hessian_bb(
            self,
            X,
            loc,
            scale,
    ):
        scalar_two = tf.constant(2, shape=(), dtype=self.dtype)
        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            X_minus_loc = tf.sparse.add(X, -loc)
        else:
            X_minus_loc = X - loc

        const = - tf.multiply(scalar_two,
            tf.square(
                tf.divide(
                    X_minus_loc,
                    scale
                )
            )
        )
        return const


