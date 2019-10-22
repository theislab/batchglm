import tensorflow as tf
from .external import FIMGLM, JacobianGLM, HessianGLM


class FIM(FIMGLM):

    def _weight_fim_aa(
            self,
            x,
            loc,
            scale
    ):
        const = tf.divide(scale, scale + loc)
        w = tf.multiply(loc, const)

        return w

    def _weight_fim_bb(
            self,
            x,
            loc,
            scale
    ):
        return tf.zeros_like(scale)


class Jacobian(JacobianGLM):

    def _weights_jac_a(
            self,
            x,
            loc,
            scale,
    ):
        if isinstance(x, tf.SparseTensor):  # or isinstance(x, tf.SparseTensorValue):
            const = tf.sparse.add(x, tf.negative(loc))
        else:
            const = tf.subtract(x, loc)
        return tf.divide(tf.multiply(scale, const), tf.add(loc, scale))

    def _weights_jac_b(self, x, loc, scale):
        # Pre-define sub-graphs that are used multiple times:
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        if isinstance(x, tf.SparseTensor):  # or isinstance(x, tf.SparseTensorValue):
            scale_plus_x = tf.sparse.add(x, scale)
        else:
            scale_plus_x = scale + x

        r_plus_mu = scale + loc

        # Define graphs for individual terms of constant term of hessian:
        const1 = tf.subtract(
            tf.math.digamma(x=scale_plus_x),
            tf.math.digamma(x=scale)
        )
        const2 = tf.negative(scale_plus_x / r_plus_mu)
        const3 = tf.add(
            tf.math.log(scale),
            scalar_one - tf.math.log(r_plus_mu)
        )
        const = tf.add_n([const1, const2, const3])  # [observations, features]
        const = scale * const

        return const


class Hessian(HessianGLM):

    def _weight_hessian_ab(self, x, loc, scale):

        if isinstance(x, tf.SparseTensor):
            x_minus_mu = tf.sparse.add(x, -loc)
        else:
            x_minus_mu = x - loc

        const = tf.multiply(
            loc * scale,
            tf.divide(
                x_minus_mu,
                tf.square(loc + scale)
            )
        )

        return const

    def _weight_hessian_aa(
            self,
            x,
            loc,
            scale,
    ):
        if isinstance(x, tf.SparseTensor):# or isinstance(x, tf.SparseTensorValue):
            x_by_scale_plus_one = tf.sparse.add(x.__div__(scale), tf.ones_like(scale))
        else:
            x_by_scale_plus_one = x / scale + tf.ones_like(scale)

        const = tf.negative(tf.multiply(
            loc,
            tf.divide(
                x_by_scale_plus_one,
                tf.square((loc / scale) + tf.ones_like(loc))
            )
        ))

        return const

    def _weight_hessian_bb(
            self,
            x,
            loc,
            scale,
    ):
        if isinstance(x, tf.SparseTensor):#  or isinstance(x, tf.SparseTensorValue):
            scale_plus_x = tf.sparse.add(x, scale)
        else:
            scale_plus_x = x + scale

        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        scalar_two = tf.constant(2, shape=(), dtype=self.dtype)
        # Pre-define sub-graphs that are used multiple times:
        scale_plus_loc = scale + loc
        # Define graphs for individual terms of constant term of hessian:
        const1 = tf.add(
            tf.math.digamma(x=scale_plus_x),
            scale * tf.math.polygamma(a=scalar_one, x=scale_plus_x)
        )
        const2 = tf.negative(tf.add(
            tf.math.digamma(x=scale),
            scale * tf.math.polygamma(a=scalar_one, x=scale)
        ))
        const3 = tf.negative(tf.divide(
            tf.add(
                loc * scale_plus_x,
                scalar_two * scale * scale_plus_loc
            ),
            tf.square(scale_plus_loc)
        ))
        const4 = tf.add(
            tf.math.log(scale),
            scalar_two - tf.math.log(scale_plus_loc)
        )
        const = tf.add_n([const1, const2, const3, const4])
        const = tf.multiply(scale, const)
        return const
