import abc
import logging

import tensorflow as tf

from .model import ModelVarsGLM

logger = logging.getLogger(__name__)


class FIMGLM:
    """
    Compute expected fisher information matrix (FIM)
    for iteratively re-weighted least squares (IWLS or IRLS) parameter updates for GLMs.
    """

    @abc.abstractmethod
    def fim_a_analytic(
            self,
            model
    ):
        pass

    @abc.abstractmethod
    def fim_b_analytic(
            self,
            model
    ):
        pass

    @abc.abstractmethod
    def _weight_fim_aa(
            self,
            mu,
            r
    ):
        """
        Compute for mean model IWLS update for a GLM.

        :param mu: tf.tensor observations x features
           Value of mean model by observation and feature.
        :param r: tf.tensor observations x features
           Value of dispersion model by observation and feature.

        :return tuple of tf.tensors
           Constants with respect to coefficient index for
           Fisher information matrix and score function computation.
        """
        pass

    @abc.abstractmethod
    def _weight_fim_bb(
            self,
            X,
            mu,
            r
    ):
        """
        Compute for dispersion model IWLS update for a GLM.

        :param X: tf.tensor observations x features
            Observation by observation and feature.
        :param mu: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param r: tf.tensor observations x features
            Value of dispersion model by observation and feature.
        :param log_rr: tf.tensor observations x features
            Logarithm of dispersion model by observation and feature.

        :return tuple of tf.tensors
            Constants with respect to coefficient index for
            Fisher information matrix and score function computation.
        """
        pass