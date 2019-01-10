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

    _update_a: bool
    _update_b: bool

    theta_new: tf.Tensor
    delta_theta_a: tf.Tensor
    delta_theta_b: tf.Tensor

    def __init__(
            self,
            batched_data: tf.data.Dataset,
            sample_indices: tf.Tensor,
            constraints_loc,
            constraints_scale,
            model_vars: ModelVarsGLM,
            noise_model: str,
            dtype,
            mode="obs",
            iterator=True,
            update_a=True,
            update_b=True,
    ):
        """ Return computational graph for hessian based on mode choice.

        :param batched_data:
            Dataset iterator over mini-batches of data (used for training) or tf.Tensors of mini-batch.
        :param sample_indices: Indices of samples to be used.
        :param constraints_loc: np.ndarray (constraints on mean model x mean model parameters)
            Constraints for location model.
            Array with constraints in rows and model parameters in columns.
            Each constraint contains non-zero entries for the a of parameters that
            has to sum to zero. This constraint is enforced by binding one parameter
            to the negative sum of the other parameters, effectively representing that
            parameter as a function of the other parameters. This dependent
            parameter is indicated by a -1 in this array, the independent parameters
            of that constraint (which may be dependent at an earlier constraint)
            are indicated by a 1.
        :param constraints_scale: np.ndarray (constraints on mean model x mean model parameters)
            Constraints for scale model.
            Array with constraints in rows and model parameters in columns.
            Each constraint contains non-zero entries for the a of parameters that
            has to sum to zero. This constraint is enforced by binding one parameter
            to the negative sum of the other parameters, effectively representing that
            parameter as a function of the other parameters. This dependent
            parameter is indicated by a -1 in this array, the independent parameters
            of that constraint (which may be dependent at an earlier constraint)
            are indicated by a 1.
        :param model_vars: TODO
        :param noise_model: str {"nb"}
            Noise model identifier.
        :param dtype: Precision used in tensorflow.
        :param mode: str
            Mode by with which hessian is to be evaluated,
            for analytic solutions of the hessian one can either chose by
            "feature" or by "obs" (observation). Note that sparse
            observation matrices X are often csr, ie. slicing is
            faster by row/observation, so that hessian evaluation
            by observation is much faster. "tf" allows for
            evaluation of the hessian via the tf.hessian function,
            which is done by feature for implementation reasons.
        :param iterator: bool
            Whether batched_data is an iterator or a tensor (such as single yield of an iterator).
        :param update_a: bool
            Wether to compute IWLS updates for a parameters.
        :param update_b: bool
            Wether to compute IWLS updates for b parameters.
        """
        self.noise_model = noise_model
        self._update_a = update_a
        self._update_b = update_b

        fim_a, fim_b = self.analytic(
            batched_data=batched_data,
            sample_indices=sample_indices,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            model_vars=model_vars,
            iterator=iterator,
            dtype=dtype
        )

        self.fim_a = fim_a
        self.fim_b = fim_b

    @abc.abstractmethod
    def analytic(
            self,
            batched_data,
            sample_indices,
            constraints_loc,
            constraints_scale,
            model_vars: ModelVarsGLM,
            iterator,
            dtype
    ):
        pass

    @abc.abstractmethod
    def _W_aa(
            self,
            mu,
            r
    ):
        """
        Compute for mean model IWLS update for a GLM.

        :param X: tf.tensor observations x features
           Observation by observation and feature.
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
    def _W_bb(
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