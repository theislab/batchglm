import logging

import tensorflow as tf

from .external import ModelVarsGLM, JacobiansGLM

logger = logging.getLogger(__name__)


class JacobiansGLMALL(JacobiansGLM):
    """
    Compute the Jacobian matrix for a GLM using gradients from tensorflow.
    """

    def jac_analytic(
            self,
            model
    ) -> tf.Tensor:
        """
        Compute the closed-form of the base_glm_all model jacobian
        by evalutating its terms grouped by observations.
        """

        def _a_byobs(X, design_loc, mu, r):
            """
            Compute the mean model block of the jacobian.

            :param X: tf.tensor observations x features
                Observation by observation and feature.
            :param mu: tf.tensor observations x features
                Value of mean model by observation and feature.
            :param r: tf.tensor observations x features
                Value of dispersion model by observation and feature.
            :return Jblock: tf.tensor features x coefficients
                Block of jacobian.
            """
            W = self._weights_jac_a(X=X, mu=mu, r=r)  # [observations, features]
            if self.constraints_loc is not None:
                XH = tf.matmul(design_loc, self.constraints_loc)
            else:
                XH = design_loc

            Jblock = tf.matmul(tf.transpose(W), XH)  # [features, coefficients]
            return Jblock

        def _b_byobs(X, design_scale, mu, r):
            """
            Compute the dispersion model block of the jacobian.
            """
            W = self._weights_jac_b(X=X, mu=mu, r=r)  # [observations, features]
            if self.constraints_scale is not None:
                XH = tf.matmul(design_scale, self.constraints_scale)
            else:
                XH = design_scale

            Jblock = tf.matmul(tf.transpose(W), XH)  # [features, coefficients]
            return Jblock

        if self.compute_a and self.compute_b:
            J_a = _a_byobs(X=model.X, design_loc=model.design_loc, mu=model.mu, r=model.r)
            J_b = _b_byobs(X=model.X, design_scale=model.design_scale, mu=model.mu, r=model.r)
            J = tf.concat([J_a, J_b], axis=1)
        elif self.compute_a and not self.compute_b:
            J = _a_byobs(X=model.X, design_loc=model.design_loc, mu=model.mu, r=model.r)
        elif not self.compute_a and self.compute_b:
            J = _b_byobs(X=model.X, design_scale=model.design_scale, mu=model.mu, r=model.r)
        else:
            raise ValueError("either require train_a or train_b")

        return J

    def jac_tf(
            self,
            model
    ) -> tf.Tensor:
        """
        Compute the Jacobian matrix for a GLM using gradients from tensorflow.
        """
        def _jac(model):
            J = tf.gradients(model.log_likelihood, self.model_vars.params)[0]
            J = tf.transpose(J)
            return J

        def _jac_a(model):
            J_a = tf.gradients(model.log_likelihood, self.model_vars.a_var)[0]
            J_a = tf.transpose(J_a)
            return J_a

        def _jac_b(model):
            J_b = tf.gradients(model.log_likelihood, self.model_vars.b_var)[0]
            J_b = tf.transpose(J_b)
            return J_b

        if self.compute_a and self.compute_b:
            J = _jac(model=model)
        elif self.compute_a and not self.compute_b:
            J = _jac_a(model=model)
        elif not self.compute_a and self.compute_b:
            J = _jac_b(model=model)
        else:
            raise ValueError("either require train_a or train_b")

        return J
