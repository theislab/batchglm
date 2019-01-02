import logging
from typing import List

import tensorflow as tf

from .model import ModelVarsGLM

logger = logging.getLogger(__name__)


class JacobiansGLM:
    """
    Wrapper to compute the Jacobian matrix for a GLM.
    """

    jac: tf.Tensor
    neg_jac: tf.Tensor

    def __init__(
            self,
            batched_data: tf.data.Dataset,
            sample_indices: tf.Tensor,
            batch_model,
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
            Dataset iterator over mini-batches of data (used for training) or tf.Tensors of mini-batch.
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
        if constraints_loc is not None and mode != "tf":
            raise ValueError("closed form jacobians do not work if constraints_loc is not None")
        if constraints_scale is not None and mode != "tf":
            raise ValueError("closed form jacobians do not work if constraints_scale is not None")

        logger.debug("jacobian mode: %s" % mode)
        logger.debug("compute jacobian for a model: %s" % str(jac_a))
        logger.debug("compute jacobian for b model: %s" % str(jac_b))

        self.noise_model = noise_model
        self._compute_jac_a = jac_a
        self._compute_jac_b = jac_b

        if mode == "analytic":
            J = self.analytic(
                batched_data=batched_data,
                sample_indices=sample_indices,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                model_vars=model_vars,
                iterator=iterator,
                dtype=dtype
            )
        elif mode == "tf":
            # Tensorflow computes the jacobian based on the objective,
            # which is the negative log-likelihood. Accordingly, the jacobian
            # is the negative jacobian computed here.
            J = self.tf(
                batched_data=batched_data,
                sample_indices=sample_indices,
                batch_model=batch_model,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                model_vars=model_vars,
                iterator=iterator,
                dtype=dtype
            )
        else:
            raise ValueError("mode not recognized in Jacobian: " + mode)

        # Assign jacobian blocks.
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
        else:
            raise ValueError("either require jac_a or jac_b")

        self.jac = J
        self.jac_a = J_a
        self.jac_b = J_b
        self.neg_jac = negJ
        self.neg_jac_a = negJ_a
        self.neg_jac_b = negJ_b

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
        raise NotImplementedError()

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
        raise NotImplementedError()