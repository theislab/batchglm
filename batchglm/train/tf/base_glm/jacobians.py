import abc
import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


class JacobiansGLM:
    """
    Compute the Jacobian matrix for a GLM.
    """

    def jac_analytic(
            self,
            model
    ) -> tf.Tensor:
        raise NotImplementedError()

    def jac_tf(
            self,
            model
    ) -> tf.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def _weights_jac_a(
            self,
            X,
            mu,
            r,
    ):
        """
        Compute the coefficient index invariant part of the
        mean model gradient.

        :param X: tf.tensor observations x features
            Observation by observation and feature.
        :param mu: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param r: tf.tensor observations x features
            Value of dispersion model by observation and feature.

        :return const: tf.tensor observations x features
            Coefficient invariant terms of hessian of
            given observations and features.
        """
        pass

    @abc.abstractmethod
    def _weights_jac_b(
            self,
            X,
            mu,
            r,
    ):
        """
        Compute the coefficient index invariant part of the
        dispersion model gradient.

        :param X: tf.tensor observations x features
            Observation by observation and feature.
        :param mu: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param r: tf.tensor observations x features
            Value of dispersion model by observation and feature.

        :return const: tf.tensor observations x features
            Coefficient invariant terms of hessian of
            given observations and features.
        """
        pass