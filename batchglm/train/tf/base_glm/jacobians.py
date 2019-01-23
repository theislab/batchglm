import abc
import logging
from typing import Union

import tensorflow as tf

from .model import ModelVarsGLM

logger = logging.getLogger(__name__)


class JacobiansGLM:
    """
    Wrapper to compute the Jacobian matrix for a GLM.
    """

    noise_model: str
    constraints_loc: tf.Tensor
    constraints_scale: tf.Tensor
    model_vars: ModelVarsGLM
    noise_model: str
    _compute_jac_a: bool
    _compute_jac_b: bool

    jac: tf.Tensor
    jac_a: Union[tf.Tensor, None]
    jac_b: Union[tf.Tensor, None]
    neg_jac: tf.Tensor
    neg_jac_a: Union[tf.Tensor, None]
    neg_jac_b: Union[tf.Tensor, None]

    def __init__(
            self,
            batched_data: tf.data.Dataset,
            sample_indices: tf.Tensor,
            constraints_loc,
            constraints_scale,
            model_vars: ModelVarsGLM,
            noise_model: str,
            dtype,
            mode="analytic",
            iterator=False,
            jac_a=True,
            jac_b=True
    ):
        """ Return computational graph for jacobian based on mode choice.

        :param batched_data:
            Dataset iterator over mini-batches of data (used for training) or tf.Tensor of mini-batch.
        :param sample_indices:
            Indices of samples to be used.
        :param batch_model: BasicModelGraph instance
            Allows evaluation of jacobian via tf.gradients as it contains model graph.
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
        :param dtype: Precision used in tensorflow.
        :param mode: str
            Mode by with which hessian is to be evaluated,
            "analytic" uses a closed form solution of the jacobian,
            "tf" allows for evaluation of the jacobian via the tf.gradients function.
        :param iterator: bool
            Whether an iterator or a tensor (single yield of an iterator) is given
            in.
        :param jac_a: bool
            Wether to compute Jacobian for a parameters. If both jac_a and jac_b are true,
            the entire jacobian is computed in self.jac.
        :param jac_b: bool
            Wether to compute Jacobian for b parameters. If both jac_a and jac_b are true,
            the entire jacobian is computed in self.jac.
        """
        assert jac_a or jac_b

        self.noise_model = noise_model
        self.constraints_loc = constraints_loc
        self.constraints_scale = constraints_scale
        self.model_vars = model_vars
        self.dtype = dtype
        self._compute_jac_a = jac_a
        self._compute_jac_b = jac_b

        if mode == "analytic":
            map_fun_base = self.analytic
        elif mode == "tf":
            map_fun_base = self.tf
        else:
            raise ValueError("mode %s not recognized" % mode)

        def map_fun(idx, data):
            return map_fun_base(
                sample_indices=idx,
                batched_data=data,
            )

        def init_fun():
            if self._compute_jac_a and self._compute_jac_b:
                return tf.zeros([model_vars.n_features, model_vars.params.shape[0]], dtype=dtype)
            elif self._compute_jac_a and not self._compute_jac_b:
                return tf.zeros([model_vars.n_features, model_vars.a_var.shape[0]], dtype=dtype)
            elif not self._compute_jac_a and self._compute_jac_b:
                return tf.zeros([model_vars.n_features, model_vars.b_var.shape[0]], dtype=dtype)

        def reduce_fun(old, new):
            return tf.add(old, new)

        if iterator:
            # Perform a reduction operation across data set.
            J = batched_data.reduce(
                initial_state=init_fun(),
                reduce_func=lambda old, new: reduce_fun(old, map_fun(new[0], new[1]))
            )
        else:
            # Only evaluate Jacobian for given data batch.
            J = map_fun(
                idx=sample_indices,
                data=batched_data
            )

        # Assign Jacobian blocks.
        p_shape_a = model_vars.a_var.shape[0]  # This has to be _var to work with constraints.
        if self._compute_jac_a and self._compute_jac_b:
            J_a = J[:, :p_shape_a]
            J_b = J[:, p_shape_a:]
            negJ = tf.negative(J)
            negJ_a = tf.negative(J_a)
            negJ_b = tf.negative(J_b)
        elif self._compute_jac_a and not self._compute_jac_b:
            J_a = J
            J_b = None
            J = J
            negJ = tf.negative(J)
            negJ_a = tf.negative(J_a)
            negJ_b = None
        elif not self._compute_jac_a and self._compute_jac_b:
            J_a = None
            J_b = J
            J = J
            negJ = tf.negative(J)
            negJ_a = None
            negJ_b = tf.negative(J_b)

        self.jac = J
        self.jac_a = J_a
        self.jac_b = J_b
        self.neg_jac = negJ
        self.neg_jac_a = negJ_a
        self.neg_jac_b = negJ_b

    def analytic(
            self,
            sample_indices,
            batched_data
    ) -> tf.Tensor:
        raise NotImplementedError()

    def tf(
            self,
            sample_indices,
            batched_data
    ) -> tf.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def _W_a(
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
    def _W_b(
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