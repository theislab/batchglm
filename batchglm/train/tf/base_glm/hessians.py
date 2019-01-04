import abc
import logging
from typing import List

import tensorflow as tf

from .model import ModelVarsGLM

logger = logging.getLogger(__name__)


class HessiansGLM:
    """
    Wrapper to compute the Hessian matrix for a GLM.
    """

    hessian: tf.Tensor
    neg_hessian: tf.Tensor

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
            hess_a=True,
            hess_b=True,
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
        :param hess_a: bool
            Wether to compute Hessian block for a parameters. If both hess_a and hess_b are true,
            the entire hessian with the off-diagonal a-b block is computed in self.hessian.
        :param hess_b: bool
            Wether to compute Hessian block for b parameters. If both hess_a and hess_b are true,
            the entire hessian with the off-diagonal a-b block is computed in self.hessian.
        """
        self.noise_model = noise_model
        self._compute_hess_a = hess_a
        self._compute_hess_b = hess_b

        if mode == "obs_batched":
            H = self.byobs(
                batched_data=batched_data,
                sample_indices=sample_indices,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                model_vars=model_vars,
                iterator=iterator,
                dtype=dtype
            )
        elif mode == "feature":
            H = self.byfeature(
                batched_data=batched_data,
                sample_indices=sample_indices,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                model_vars=model_vars,
                iterator=iterator,
                dtype=dtype
            )
        elif mode == "tf":
            H = self.tf_byfeature(
                batched_data=batched_data,
                sample_indices=sample_indices,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                model_vars=model_vars,
                iterator=iterator,
                dtype=dtype
            )
        else:
            raise ValueError("mode %s not recognized" % mode)

        # Assign jacobian blocks.
        p_shape_a = model_vars.a_var.shape[0]  # This has to be _var to work with constraints.
        if self._compute_hess_a and self._compute_hess_b:
            H_a = H[:, :p_shape_a, :p_shape_a]
            H_b = H[:, p_shape_a:, p_shape_a:]
            negH = tf.negative(H)
            negH_a = tf.negative(H_a)
            negH_b = tf.negative(H_b)
        elif self._compute_hess_a and not self._compute_hess_b:
            H_a = H
            H_b = None
            negH = tf.negative(H)
            negH_a = tf.negative(H_a)
            negH_b = None
        elif not self._compute_hess_a and self._compute_hess_b:
            H_a = None
            H_b = H
            negH = tf.negative(H)
            negH_a = None
            negH_b = tf.negative(H_b)
        else:
            raise ValueError("either require jac_a or jac_b")

        self.hessian = H
        self.hessian_aa = H_a
        self.hessian_bb = H_b
        self.neg_hessian = negH
        self.neg_hessian_aa = negH_a
        self.neg_hessian_bb = negH_b

    def byobs(
            self,
            batched_data,
            sample_indices,
            constraints_loc,
            constraints_scale,
            model_vars: ModelVarsGLM,
            iterator,
            dtype
    ):
        raise NotImplementedError()

    def byfeature(
            self,
            batched_data,
            sample_indices,
            constraints_loc,
            constraints_scale,
            model_vars: ModelVarsGLM,
            iterator,
            dtype
    ):
        raise NotImplementedError()

    def tf_byfeature(
            self,
            batched_data,
            sample_indices,
            constraints_loc,
            constraints_scale,
            model_vars: ModelVarsGLM,
            iterator,
            dtype
    ) -> List[tf.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _W_aa(
            self,
            X,
            mu,
            r,
    ):
        """
        Compute the coefficient index invariant part of the
        mean model block of the hessian.

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
    def _W_bb(
            self,
            X,
            mu,
            r,
    ):
        """
        Compute the coefficient index invariant part of the
        dispersion model block of the hessian.

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
    def _W_ab(
            self,
            X,
            mu,
            r,
    ):
        """
        Compute the coefficient index invariant part of the
        mean-dispersion model block of the hessian.

        Note that there are two blocks of the same size which can
        be compute from each other with a transpose operation as
        the hessian is symmetric.

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

