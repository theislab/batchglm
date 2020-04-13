import abc
import tensorflow as tf


class Gradient(tf.keras.layers.Layer):

    """Superclass for Jacobians, Hessian, FIM"""

    def __init__(self, model_vars, dtype):
        super(Gradient, self).__init__()
        self.model_vars = model_vars
        self.grad_dtype = dtype

    @abc.abstractmethod
    def call(self, inputs, **kwargs):
        pass

    @staticmethod
    def calc_design_mat(design_mat, constraints):
        if constraints is not None:
            xh = tf.matmul(design_mat, constraints)
        else:
            xh = design_mat
        return xh

    # Here, we use the einsum to efficiently perform the two outer products and the marginalisation.
    @staticmethod
    def create_specific_block(w, xh_loc, xh_scale):
        return tf.einsum('ofc,od->fcd', tf.einsum('of,oc->ofc', w, xh_loc), xh_scale)


class FIMGLM(Gradient):
    """
    Compute expected fisher information matrix (FIM)
    for iteratively re-weighted least squares (IWLS or IRLS) parameter updates for GLMs.
    """

    def call(self, inputs, **kwargs):
        return self._fim_analytic(*inputs)

    def _fim_analytic(self, x, design_loc, design_scale, loc, scale, concat=False, compute_a=True, compute_b=True) -> tf.Tensor:
        """
        Compute the closed-form of the base_glm_all model fim
        by evalutating its terms grouped by observations.
        """

        def _a_byobs():
            """
            Compute the mean model diagonal block of the
            closed form fim of base_glm_all model by observation across features
            for a batch of observations.
            """
            w = self._weight_fim_aa(x=x, loc=loc, scale=scale)  # [observations x features]
            # The computation of the fim block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the einsum to efficiently perform the two outer products and the marginalisation.
            xh = self.calc_design_mat(design_loc, self.model_vars.constraints_loc)

            fim_block = self.create_specific_block(w, xh, xh)
            return fim_block

        def _b_byobs():
            """
            Compute the dispersion model diagonal block of the
            closed form fim of base_glm_all model by observation across features.
            """
            w = self._weight_fim_bb(x=x, loc=loc, scale=scale)  # [observations=1 x features]
            # The computation of the fim block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the Einstein summation to efficiently perform the two outer products and the marginalisation.
            xh = self.calc_design_mat(design_scale, self.model_vars.constraints_scale)

            fim_block = self.create_specific_block(w, xh, xh)
            return fim_block

        # The full fisher information matrix is block-diagonal with the cross-model
        # blocks all zero. Accordingly, mean and dispersion model updates can be
        # treated independently and the full fisher information matrix is never required.
        # Here, the non-zero model-wise diagonal blocks are computed and returned
        # as a dictionary. The according score function vectors are also returned as a dictionary.

        if compute_a and compute_b:
            fim_a = _a_byobs()
            fim_b = _b_byobs()

        elif compute_a and not compute_b:
            fim_a = _a_byobs()
            fim_b = tf.zeros(fim_a.get_shape(), self.grad_dtype)
        elif not compute_a and compute_b:
            fim_a = tf.zeros(fim_a.get_shape(), self.grad_dtype)
            fim_b = _b_byobs()
        else:
            fim_a = tf.zeros_like(self.model_vars.a_var, dtype=self.grad_dtype)
            fim_b = tf.zeros_like(self.model_vars.b_var, dtype=self.grad_dtype)

        if concat:
            fim = tf.concat([fim_a, fim_b], axis=1)
            return fim
        else:
            return fim_a, fim_b

    @abc.abstractmethod
    def _weight_fim_aa(
            self,
            x,
            loc,
            scale
    ):
        """
        Compute for mean model IWLS update for a GLM.

        :param loc: tf.tensor observations x features
           Value of mean model by observation and feature.
        :param scale: tf.tensor observations x features
           Value of dispersion model by observation and feature.

        :return tuple of tf.tensors
           Constants with respect to coefficient index for
           Fisher information matrix and score function computation.
        """
        pass

    @abc.abstractmethod
    def _weight_fim_bb(
            self,
            x,
            loc,
            scale
    ):
        """
        Compute for dispersion model IWLS update for a GLM.

        :param x: tf.tensor observations x features
            Observation by observation and feature.
        :param loc: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param scale: tf.tensor observations x features
            Value of dispersion model by observation and feature.

        :return tuple of tf.tensors
            Constants with respect to coefficient index for
            Fisher information matrix and score function computation.
        """
        pass


class JacobianGLM(Gradient):

    def call(self, inputs, **kwargs):
        return self._jac_analytic(*inputs)

    def _jac_analytic(self, x, design_loc, design_scale, loc, scale, concat, compute_a=True, compute_b=True) -> tf.Tensor:
        """
        Compute the closed-form of the base_glm_all model jacobian
        by evalutating its terms grouped by observations.

        :param x: tf.tensor observations x features
                Observation by observation and feature.
        :param loc: tf.tensor observations x features
                Value of mean model by observation and feature.
        :param scale: tf.tensor observations x features
                Value of dispersion model by observation and feature.
        """

        def _a_byobs():
            """
            Compute the mean model block of the jacobian.

            :return Jblock: tf.tensor features x coefficients
                Block of jacobian.
            """
            w = self._weights_jac_a(x=x, loc=loc, scale=scale)  # [observations, features]
            xh = self.calc_design_mat(design_loc, self.model_vars.constraints_loc)  # [observations, coefficient]

            jblock = tf.matmul(tf.transpose(w), xh)  # [features, coefficients]
            return jblock

        def _b_byobs():
            """
            Compute the dispersion model block of the jacobian.

            :return Jblock: tf.tensor features x coefficients
                Block of jacobian.
            """
            w = self._weights_jac_b(x=x, loc=loc, scale=scale)  # [observations, features]
            xh = self.calc_design_mat(design_scale, self.model_vars.constraints_scale)  # [observations, coefficient]

            jblock = tf.matmul(tf.transpose(w), xh)  # [features, coefficients]
            return jblock

        if compute_a and compute_b:
            j_a = _a_byobs()
            j_b = _b_byobs()
        elif compute_a and not compute_b:
            j_a = _a_byobs()
            j_b = tf.zeros((j_a.get_shape()[0], self.model_vars.b_var.get_shape()[0]), dtype=self.grad_dtype)
        elif not compute_a and compute_b:
            j_b = _b_byobs()
            j_a = tf.zeros((j_b.get_shape()[0], self.model_vars.b_var.get_shape()[0]), dtype=self.grad_dtype)
        else:
            j_a = tf.transpose(tf.zeros_like(self.model_vars.a_var, dtype=self.grad_dtype))
            j_b = tf.transpose(tf.zeros_like(self.model_vars.b_var, dtype=self.grad_dtype))

        if concat:
            j = tf.concat([j_a, j_b], axis=1)
            return j
        else:
            return j_a, j_b

    @abc.abstractmethod
    def _weights_jac_a(
            self,
            x,
            loc,
            scale
    ):
        """
        Compute the coefficient index invariant part of the
        mean model gradient.

        :param x: tf.tensor observations x features
            Observation by observation and feature.
        :param loc: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param scale: tf.tensor observations x features
            Value of dispersion model by observation and feature.

        :return const: tf.tensor observations x features
            Coefficient invariant terms of hessian of
            given observations and features.
        """
        pass

    @abc.abstractmethod
    def _weights_jac_b(
            self,
            x,
            loc,
            scale
    ):
        """
        Compute the coefficient index invariant part of the
        dispersion model gradient.

        :param x: tf.tensor observations x features
            Observation by observation and feature.
        :param loc: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param scale: tf.tensor observations x features
            Value of dispersion model by observation and feature.

        :return const: tf.tensor observations x features
            Coefficient invariant terms of hessian of
            given observations and features.
        """
        pass


class HessianGLM(Gradient):
    """
    Compute the closed-form of the base_glm_all model hessian
    by evaluating its terms grouped by observations.

    Has three sub-functions which built the specific blocks of the hessian
    and one sub-function which concatenates the blocks into a full hessian.
    """

    def call(self, inputs, **kwargs):
        return self._hessian_analytic(*inputs)

    def _hessian_analytic(self, x, design_loc, design_scale, loc, scale, concat, compute_a=True, compute_b=True) -> tf.Tensor:
        """
        Compute the closed-form of the base_glm_all model hessian
        by evaluating its terms grouped by observations.

        Has three sub-functions which built the specific blocks of the hessian
        and one sub-function which concatenates the blocks into a full hessian.
        """

        def _aa_byobs_batched():
            """
            Compute the mean model diagonal block of the
            closed form hessian of base_glm_all model by observation across features
            for a batch of observations.
            """
            w = self._weight_hessian_aa(x=x, loc=loc, scale=scale)  # [observations x features]
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the einsum to efficiently perform the two outer products and the marginalisation.
            xh = self.calc_design_mat(design_loc, self.model_vars.constraints_loc)

            hblock = self.create_specific_block(w, xh, xh)
            return hblock

        def _bb_byobs_batched():
            """
            Compute the dispersion model diagonal block of the
            closed form hessian of base_glm_all model by observation across features.
            """
            w = self._weight_hessian_bb(x=x, loc=loc, scale=scale)  # [observations x features]
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the Einstein summation to efficiently perform the two outer products and the marginalisation.
            xh = self.calc_design_mat(design_scale, self.model_vars.constraints_scale)

            hblock = self.create_specific_block(w, xh, xh)
            return hblock

        def _ab_byobs_batched():
            """
            Compute the mean-dispersion model off-diagonal block of the
            closed form hessian of base_glm_all model by observastion across features.

            Note that there are two blocks of the same size which can
            be compute from each other with a transpose operation as
            the hessian is symmetric.
            """
            w = self._weight_hessian_ab(x=x, loc=loc, scale=scale)  # [observations x features]
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the Einstein summation to efficiently perform the two outer products and the marginalisation.
            xhloc = self.calc_design_mat(design_loc, self.model_vars.constraints_loc)
            xhscale = self.calc_design_mat(design_scale, self.model_vars.constraints_scale)

            hblock = self.create_specific_block(w, xhloc, xhscale)
            return hblock

        if compute_a and compute_b:
            h_aa = _aa_byobs_batched()
            h_bb = _bb_byobs_batched()
            h_ab = _ab_byobs_batched()
            h_ba = tf.transpose(h_ab, perm=[0, 2, 1])
        elif compute_a and not compute_b:
            h_aa = _aa_byobs_batched()
            h_bb = tf.zeros_like(h_aa, dtype=self.grad_dtype)
            h_ab = tf.zeros_like(h_aa, dtype=self.grad_dtype)
            h_ba = tf.zeros_like(h_aa, dtype=self.grad_dtype)
        elif not compute_a and compute_b:
            h_bb = _bb_byobs_batched()
            h_aa = tf.zeros_like(h_bb, dtype=self.grad_dtype)
            h_ab = tf.zeros_like(h_bb, dtype=self.grad_dtype)
            h_ba = tf.zeros_like(h_bb, dtype=self.grad_dtype)
        else:
            h_aa = tf.zeros((), dtype=self.grad_dtype)
            h_bb = tf.zeros((), dtype=self.grad_dtype)
            h_ab = tf.zeros((), dtype=self.grad_dtype)
            h_ba = tf.zeros((), dtype=self.grad_dtype)

        if concat:
            h = tf.concat(
                [tf.concat([h_aa, h_ab], axis=2),
                 tf.concat([h_ba, h_bb], axis=2)],
                axis=1
            )
            return h
        else:
            return h_aa, h_ab, h_ba, h_bb

    @abc.abstractmethod
    def _weight_hessian_aa(
            self,
            x,
            loc,
            scale
    ):
        """
        Compute the coefficient index invariant part of the
        mean model block of the hessian.

        :param x: tf.tensor observations x features
            Observation by observation and feature.
        :param loc: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param scale: tf.tensor observations x features
            Value of dispersion model by observation and feature.

        :return const: tf.tensor observations x features
            Coefficient invariant terms of hessian of
            given observations and features.
        """
        pass

    @abc.abstractmethod
    def _weight_hessian_bb(
            self,
            x,
            loc,
            scale
    ):
        """
        Compute the coefficient index invariant part of the
        dispersion model block of the hessian.

        :param x: tf.tensor observations x features
            Observation by observation and feature.
        :param loc: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param scale: tf.tensor observations x features
            Value of dispersion model by observation and feature.

        :return const: tf.tensor observations x features
            Coefficient invariant terms of hessian of
            given observations and features.
        """
        pass

    @abc.abstractmethod
    def _weight_hessian_ab(
            self,
            x,
            loc,
            scale
    ):
        """
        Compute the coefficient index invariant part of the
        mean-dispersion model block of the hessian.

        Note that there are two blocks of the same size which can
        be compute from each other with a transpose operation as
        the hessian is symmetric.

        :param x: tf.tensor observations x features
            Observation by observation and feature.
        :param loc: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param scale: tf.tensor observations x features
            Value of dispersion model by observation and feature.

        :return const: tf.tensor observations x features
            Coefficient invariant terms of hessian of
            given observations and features.
        """
        pass
