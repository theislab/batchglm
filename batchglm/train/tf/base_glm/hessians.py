import abc
import logging
from typing import Tuple, Union

import tensorflow as tf

from .model import ModelVarsGLM

logger = logging.getLogger(__name__)


class HessiansGLM:
    """
    Wrapper to compute the Hessian matrix for a GLM.
    """

    noise_model: str
    constraints_loc: tf.Tensor
    constraints_scale: tf.Tensor
    model_vars: ModelVarsGLM
    _compute_hess_a: bool
    _compute_hess_b: bool

    hessian: tf.Tensor
    hessian_aa: Union[tf.Tensor, None]
    hessian_bb: Union[tf.Tensor, None]
    neg_hessian: tf.Tensor
    neg_hessian_aa: Union[tf.Tensor, None]
    neg_hessian_bb: Union[tf.Tensor, None]

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
        self.constraints_loc = constraints_loc
        self.constraints_scale = constraints_scale
        self.model_vars = model_vars
        self.dtype = dtype
        self._compute_hess_a = hess_a
        self._compute_hess_b = hess_b

        def init_fun():
            if self._compute_hess_a and self._compute_hess_b:
                return tf.zeros([model_vars.n_features,
                                 model_vars.params.shape[0],
                                 model_vars.params.shape[0]], dtype=dtype)
            elif self._compute_hess_a and not self._compute_hess_b:
                return tf.zeros([model_vars.n_features,
                                 model_vars.a_var.shape[0],
                                 model_vars.a_var.shape[0]], dtype=dtype)
            elif not self._compute_hess_a and self._compute_hess_b:
                return tf.zeros([model_vars.n_features,
                                 model_vars.b_var.shape[0],
                                 model_vars.b_var.shape[0]], dtype=dtype)

        if mode == "obs_batched":
            map_fun_base = self.byobs
        elif mode == "feature":
            map_fun_base = self.byfeature
        elif mode == "tf":
            map_fun_base = self.tf_byfeature
        else:
            raise ValueError("mode %s not recognized" % mode)

        def map_fun(idx, data):
            return map_fun_base(
                sample_indices=idx,
                batched_data=data
            )

        def reduce_fun(old, new):
            return tf.add(old, new)

        if iterator:
            # Perform a reduction operation across data set.
            H = batched_data.reduce(
                initial_state=init_fun(),
                reduce_func=lambda old, new: reduce_fun(old, map_fun(new[0], new[1]))
            )
        else:
            # Only evaluate Hessian for given data batch.
            H = map_fun(
                idx=sample_indices,
                data=batched_data
            )

        # Assign jacobian blocks.
        p_shape_a = model_vars.a_var.shape[0]  # This has to be _var to work with constraints.

        # With relay across tf.Variable:
        #if self._compute_hess_a and self._compute_hess_b:
        #    self.hessian = tf.Variable(tf.zeros([model_vars.n_features,
        #                                         model_vars.params.shape[0],
        #                                         model_vars.params.shape[0]], dtype=dtype), dtype=dtype)
        #    self.hessian_aa = self.hessian[:, :p_shape_a, :p_shape_a]
        #    self.hessian_bb = self.hessian[:, p_shape_a:, p_shape_a:]
        #elif self._compute_hess_a and not self._compute_hess_b:
        #    self.hessian = tf.Variable(tf.zeros([model_vars.n_features,
        #                                         model_vars.a_var.shape[0],
        #                                         model_vars.a_var.shape[0]], dtype=dtype), dtype=dtype)
        #    self.hessian_aa = self.hessian
        #    self.hessian_bb = None
        #elif not self._compute_hess_a and self._compute_hess_b:
        #    self.hessian = tf.Variable(tf.zeros([model_vars.n_features,
        #                                         model_vars.b_var.shape[0],
        #                                         model_vars.b_var.shape[0]], dtype=dtype), dtype=dtype)
        #    self.hessian_aa = self.hessian
        #    self.hessian_bb = None
        #else:
        #    raise ValueError("either require jac_a or jac_b")
        # Save as variables:
        #self.hessian_set = tf.assign(self.hessian, H)

        # Without relay across tf.Variable:
        self.hessian = H
        if self._compute_hess_a and self._compute_hess_b:
            self.hessian_aa = H[:, :p_shape_a, :p_shape_a]
            self.hessian_bb = H[:, p_shape_a:, p_shape_a:]
        elif self._compute_hess_a and not self._compute_hess_b:
            self.hessian_aa = H
            self.hessian_bb = None
        elif not self._compute_hess_a and self._compute_hess_b:
            self.hessian_aa = None
            self.hessian_bb = H
        else:
            self.hessian_aa = None
            self.hessian_bb = None

        self.neg_hessian = tf.negative(self.hessian) if self.hessian is not None else None
        self.neg_hessian_aa = tf.negative(self.hessian_aa) if self.hessian_aa is not None else None
        self.neg_hessian_bb = tf.negative(self.hessian_bb) if self.hessian_bb is not None else None

    def byobs(
            self,
            sample_indices,
            batched_data
    ) -> Tuple:
        raise NotImplementedError()

    def byfeature(
            self,
            sample_indices,
            batched_data
    ) -> Tuple:
        raise NotImplementedError()

    def tf_byfeature(
            self,
            sample_indices,
            batched_data
    ) -> Tuple:
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

