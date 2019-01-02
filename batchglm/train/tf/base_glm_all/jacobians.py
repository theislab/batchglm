import logging
from typing import List

import tensorflow as tf

from .external import ModelVarsGLM
from .external import op_utils
from .external import pkg_constants

logger = logging.getLogger(__name__)


class JacobiansTF:
    """
    Compute the Jacobian matrix for a GLM using gradients from tensorflow.
    """

    noise_model: str
    _compute_jac_a: bool
    _compute_jac_b: bool

    def tf(
            self,
            batched_data,
            sample_indices,
            batch_model,
            constraints_loc,
            constraints_scale,
            model_vars: ModelVarsGLM,
            iterator,
            dtype
    ) -> List[tf.Tensor]:
        """
        Compute the Jacobian matrix for a GLM using gradients from tensorflow.
        """

        if self.noise_model == "nb":
            from .external_nb import BasicModelGraph
        else:
            raise ValueError("noise model %s was not recognized" % self.noise_model)

        def _jac(batch_model, model_vars):
            J = tf.gradients(batch_model.log_likelihood, model_vars.params)[0]
            J = tf.transpose(J)
            return J

        def _jac_a(batch_model, model_vars):
            J_a = tf.gradients(batch_model.log_likelihood, model_vars.a_var)[0]
            J_a = tf.transpose(J_a)
            return J_a

        def _jac_b(batch_model, model_vars):
            J_b = tf.gradients(batch_model.log_likelihood, model_vars.b_var)[0]
            J_b = tf.transpose(J_b)
            return J_b

        def _assemble_bybatch(idx, data):
            """
            Assemble jacobian of a batch of observations across all features.

            This function runs the data batch (an observation) through the
            model graph and calls the wrappers that compute the
            individual closed forms of the jacobian.

            :param data: tuple
                Containing the following parameters:
                - X: tf.tensor observations x features
                    Observation by observation and feature.
                - size_factors: tf.tensor observations x features
                    Model size factors by observation and feature.
                - params: tf.tensor features x coefficients
                    Estimated model variables.
            :return J: tf.tensor features x coefficients
                Jacobian evaluated on a single observation, provided in data.
            """
            X, design_loc, design_scale, size_factors = data

            model = BasicModelGraph(
                X=X,
                design_loc=design_loc,
                design_scale=design_scale,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                a=model_vars.a,
                b=model_vars.b,
                dtype=dtype,
                size_factors=size_factors
            )

            if self._compute_jac_a and self._compute_jac_b:
                J = _jac(batch_model=model, model_vars=model_vars)
            elif self._compute_jac_a and not self._compute_jac_b:
                J = _jac_a(batch_model=model, model_vars=model_vars)
            elif not self._compute_jac_a and self._compute_jac_b:
                J = _jac_b(batch_model=model, model_vars=model_vars)
            else:
                raise ValueError("either require jac_a or jac_b")
            return J

        def _red(prev, cur):
            """
            Reduction operation for jacobian computation across observation batches.

            Every evaluation of the jacobian on an observation yields a full
            jacobian matrix. This function sums over consecutive evaluations
            of this hessian so that not all seperate evluations have to be
            stored.
            """
            return tf.add(prev, cur)

        if iterator == True and batch_model is None:
            J = op_utils.map_reduce(
                last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
                data=batched_data,
                map_fn=_assemble_bybatch,
                reduce_fn=_red,
                parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
            )
        elif iterator == False and batch_model is None:
            J = _assemble_bybatch(
                idx=sample_indices,
                data=batched_data
            )
        else:
            if self._compute_jac_a and self._compute_jac_b:
                J = _jac(batch_model=batch_model, model_vars=model_vars)
            elif self._compute_jac_a and not self._compute_jac_b:
                J = _jac_a(batch_model=batch_model, model_vars=model_vars)
            elif not self._compute_jac_a and self._compute_jac_b:
                J = _jac_b(batch_model=batch_model, model_vars=model_vars)
            else:
                raise ValueError("either require jac_a or jac_b")

        return J
