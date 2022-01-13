import tensorflow as tf
from .external import FIMGLM, JacobianGLM, HessianGLM


class FIM(FIMGLM):
    # No Fisher Information Matrices due to unsolvable E[log(X)]

    def _weight_fim_aa(
            self,
            x,
            loc,
            scale
    ):
        assert False, "not implemented"

    def _weight_fim_bb(
            self,
            x,
            loc,
            scale
    ):
        assert False, "not implemented"


class Jacobian(JacobianGLM):

    def _weights_jac_a(
            self,
            x,
            loc,
            scale,
    ):
        one_minus_loc = 1 - loc
        if isinstance(x, tf.SparseTensor):
            const1 = tf.math.log(tf.sparse.add(tf.zeros_like(loc), x).__div__(-tf.sparse.add(x, -tf.ones_like(loc))))
        else:
            const1 = tf.math.log(x / (1 - x))
        const2 = - tf.math.digamma(loc * scale) + tf.math.digamma(one_minus_loc * scale) + const1
        const = const2 * scale * loc * one_minus_loc
        return const

    def _weights_jac_b(
            self,
            x,
            loc,
            scale,
    ):
        if isinstance(x, tf.SparseTensor):
            one_minus_x = - tf.sparse.add(x, -tf.ones_like(loc))
        else:
            one_minus_x = 1 - x
        one_minus_loc = 1 - loc
        const = scale * (tf.math.digamma(scale) - tf.math.digamma(loc * scale) * loc - tf.math.digamma(
            one_minus_loc * scale) * one_minus_loc + loc * tf.math.log(x) + one_minus_loc * tf.math.log(
            one_minus_x))
        return const


class Hessian(HessianGLM):

    def _weight_hessian_aa(
            self,
            x,
            loc,
            scale,
    ):
        one_minus_loc = 1 - loc
        loc_times_scale = loc * scale
        one_minus_loc_times_scale = one_minus_loc * scale

        if isinstance(x, tf.SparseTensor):
            # Using the dense matrix  of the location model to serve the correct shapes for the sparse X.
            const1 = tf.sparse.add(tf.zeros_like(loc), x).__div__(-tf.sparse.add(x, -tf.ones_like(loc)))
            # Adding tf.zeros_like(loc) is a hack to avoid bug thrown by log on sparse matrix below,
            # to_dense does not work.
        else:
            const1 = tf.math.log(x / (tf.ones_like(x) - x))

        const2 = (1 - 2 * loc) * (
                - tf.math.digamma(loc_times_scale) + tf.math.digamma(one_minus_loc_times_scale) + const1)
        const3 = loc * one_minus_loc_times_scale * (
                - tf.math.polygamma(tf.ones_like(loc), loc_times_scale) - tf.math.polygamma(tf.ones_like(loc),
                                                                                            one_minus_loc_times_scale))
        const = loc * one_minus_loc_times_scale * (const2 + const3)
        return const

    def _weight_hessian_ab(
            self,
            x,
            loc,
            scale,
    ):
        one_minus_loc = 1 - loc
        loc_times_scale = loc * scale
        one_minus_loc_times_scale = one_minus_loc * scale
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)

        if isinstance(x, tf.SparseTensor):
            # Using the dense matrix  of the location model to serve the correct shapes for the sparse X.
            const1 = tf.sparse.add(tf.zeros_like(loc), x).__div__(-tf.sparse.add(x, -tf.ones_like(loc)))
            # Adding tf.zeros_like(loc) is a hack to avoid bug thrown by log on sparse matrix below,
            # to_dense does not work.
        else:
            const1 = tf.math.log(x / (1 - x))

        const2 = - tf.math.digamma(loc_times_scale) + tf.math.digamma(one_minus_loc_times_scale) + const1
        const3 = scale * (- tf.math.polygamma(scalar_one, loc_times_scale) * loc + one_minus_loc * tf.math.polygamma(
            scalar_one,
            one_minus_loc_times_scale))

        const = loc * one_minus_loc_times_scale * (const2 + const3)

        return const

    def _weight_hessian_bb(
            self,
            x,
            loc,
            scale,
    ):
        one_minus_loc = 1 - loc
        loc_times_scale = loc * scale
        one_minus_loc_times_scale = one_minus_loc * scale
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)

        if isinstance(x, tf.SparseTensor):
            # Using the dense matrix  of the location model to serve the correct shapes for the sparse X.
            const1 = tf.sparse.add(tf.zeros_like(loc), x).__div__(-tf.sparse.add(x, -tf.ones_like(loc)))
            # Adding tf.zeros_like(loc) is a hack to avoid bug thrown by log on sparse matrix below,
            # to_dense does not work.
            const2 = loc * (tf.math.log(tf.sparse.add(tf.zeros_like(loc), x)) - tf.math.digamma(loc_times_scale)) \
                     - one_minus_loc * (tf.math.digamma(one_minus_loc_times_scale) + tf.math.log(const1)) \
                     + tf.math.digamma(scale)
        else:
            const1 = tf.math.log(x / (1 - x))
            const2 = loc * (tf.math.log(x) - tf.math.digamma(loc_times_scale)) \
                     - one_minus_loc * (tf.math.digamma(one_minus_loc_times_scale) + tf.math.log(const1)) \
                     + tf.math.digamma(scale)
        const3 = scale * (- tf.square(loc) * tf.math.polygamma(scalar_one, loc_times_scale)
                          + tf.math.polygamma(scalar_one, scale)
                          - tf.math.polygamma(scalar_one, one_minus_loc_times_scale) * tf.square(one_minus_loc))
        const = scale * (const2 + const3)

        return const
