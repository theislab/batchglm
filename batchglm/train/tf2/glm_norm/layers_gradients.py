import tensorflow as tf
from .external import FIMGLM, JacobianGLM, HessianGLM


class FIM(FIMGLM):

    def _weight_fim_aa(
            self,
            x,
            loc,
            scale
    ):
        w = tf.square(tf.divide(tf.ones_like(scale), scale))

        return w

    def _weight_fim_bb(
            self,
            x,
            loc,
            scale
    ):
        w = tf.constant(2, shape=loc.shape, dtype=self.dtype)

        return w


class Jacobian(JacobianGLM):

    def _weights_jac_a(
            self,
            x,
            loc,
            scale,
    ):
        if isinstance(x, tf.SparseTensor):
            const1 = tf.sparse.add(x, -loc)
            const = tf.divide(const1, tf.square(scale))
        else:
            const1 = tf.subtract(x, loc)
            const = tf.divide(const1, tf.square(scale))
        return const

    def _weights_jac_b(
            self,
            x,
            loc,
            scale,
    ):
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        if isinstance(x, tf.SparseTensor):
            const = tf.negative(scalar_one) + tf.math.square(
                tf.divide(tf.sparse.add(x, -loc), scale)
            )
        else:
            const = tf.negative(scalar_one) + tf.math.square(
                tf.divide(tf.subtract(x, loc), scale)
            )
        return const


class Hessian(HessianGLM):

    def _weight_hessian_ab(
            self,
            x,
            loc,
            scale,
    ):
        scalar_two = tf.constant(2, shape=(), dtype=self.dtype)
        if isinstance(x, tf.SparseTensor):
            x_minus_loc = tf.sparse.add(x, -loc)
        else:
            x_minus_loc = x - loc

        const = - tf.multiply(scalar_two,
                              tf.divide(
                                  x_minus_loc,
                                  tf.square(scale)
                              )
                              )
        return const

    def _weight_hessian_aa(
            self,
            x,
            loc,
            scale,
    ):
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        const = - tf.divide(scalar_one, tf.square(scale))

        return const

    def _weight_hessian_bb(
            self,
            x,
            loc,
            scale,
    ):
        scalar_two = tf.constant(2, shape=(), dtype=self.dtype)
        if isinstance(x, tf.SparseTensor):
            x_minus_loc = tf.sparse.add(x, -loc)
        else:
            x_minus_loc = x - loc

        const = - tf.multiply(
            scalar_two,
            tf.math.square(
                tf.divide(
                    x_minus_loc,
                    scale
                )
            )
        )
        return const
