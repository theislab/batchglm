import abc
import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

class HessiansGLM:
    """
    Wrapper to compute the Hessian matrix for a GLM.
    """

    def hessian_analytic(
            self,
            model
    ) -> tf.Tensor:
        raise NotImplementedError()

    def hessian_tf(
            self,
            model
    ) -> tf.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def _weight_hessian_aa(
            self,
            X,
            loc,
            scale
    ):
        """
        Compute the coefficient index invariant part of the
        mean model block of the hessian.

        :param X: tf1.tensor observations x features
            Observation by observation and feature.
        :param loc: tf1.tensor observations x features
            Value of mean model by observation and feature.
        :param scale: tf1.tensor observations x features
            Value of dispersion model by observation and feature.

        :return const: tf1.tensor observations x features
            Coefficient invariant terms of hessian of
            given observations and features.
        """
        pass

    @abc.abstractmethod
    def _weight_hessian_bb(
            self,
            X,
            loc,
            scale
    ):
        """
        Compute the coefficient index invariant part of the
        dispersion model block of the hessian.

        :param X: tf1.tensor observations x features
            Observation by observation and feature.
        :param loc: tf1.tensor observations x features
            Value of mean model by observation and feature.
        :param scale: tf1.tensor observations x features
            Value of dispersion model by observation and feature.

        :return const: tf1.tensor observations x features
            Coefficient invariant terms of hessian of
            given observations and features.
        """
        pass

    @abc.abstractmethod
    def _weight_hessian_ab(
            self,
            X,
            loc,
            scale
    ):
        """
        Compute the coefficient index invariant part of the
        mean-dispersion model block of the hessian.

        Note that there are two blocks of the same size which can
        be compute from each other with a transpose operation as
        the hessian is symmetric.

        :param X: tf1.tensor observations x features
            Observation by observation and feature.
        :param loc: tf1.tensor observations x features
            Value of mean model by observation and feature.
        :param scale: tf1.tensor observations x features
            Value of dispersion model by observation and feature.

        :return const: tf1.tensor observations x features
            Coefficient invariant terms of hessian of
            given observations and features.
        """
        pass

