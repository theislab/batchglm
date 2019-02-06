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
    noise_model: str
    constraints_loc: tf.Tensor
    constraints_scale: tf.Tensor
    model_vars: ModelVarsGLM
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
            iterator=True,
            update_a=True,  # TODO remove
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
        self.constraints_loc = constraints_loc
        self.constraints_scale = constraints_scale
        self.model_vars = model_vars
        self.dtype = dtype

        def map_fun(idx, data, return_a, return_b):
            return self.analytic(
                sample_indices=idx,
                batched_data=data,
                return_a=return_a,
                return_b=return_b
            )

        def reduce_fun(old, new):
            fim = (tf.add(old[0], new[0]), tf.add(old[1], new[1]))

            return fim

        def init_fun(return_a, return_b):
            if return_a and return_b:
                return (tf.zeros([model_vars.n_features,
                                  model_vars.a_var.shape[0],
                                  model_vars.a_var.shape[0]], dtype=dtype),
                        tf.zeros([model_vars.n_features,
                                  model_vars.b_var.shape[0],
                                  model_vars.b_var.shape[0]], dtype=dtype))
            elif return_a and not return_b:
                return (tf.zeros([model_vars.n_features,
                                  model_vars.a_var.shape[0],
                                  model_vars.a_var.shape[0]], dtype=dtype),
                        tf.zeros((), dtype=dtype))
            elif not return_a and return_b:
                return (tf.zeros((), dtype=dtype),
                        tf.zeros([model_vars.n_features,
                                  model_vars.b_var.shape[0],
                                  model_vars.b_var.shape[0]], dtype=dtype))
            else:
                assert False, "chose at least one of return_a and return_a"

        if iterator:
            # Perform a reduction operation across data set.
            fim_a = batched_data.reduce(
                initial_state=init_fun(return_a=True, return_b=False),
                reduce_func=lambda old, new: reduce_fun(old, map_fun(new[0], new[1], return_a=True, return_b=False))
            )
            fim_b = batched_data.reduce(
                initial_state=init_fun(return_a=False, return_b=True),
                reduce_func=lambda old, new: reduce_fun(old, map_fun(new[0], new[1], return_a=False, return_b=True))
            )
            fim_ab = batched_data.reduce(
                initial_state=init_fun(return_a=True, return_b=True),
                reduce_func=lambda old, new: reduce_fun(old, map_fun(new[0], new[1], return_a=True, return_b=True))
            )
        else:
            # Only evaluate FIM for given data batch.
            fim_a = map_fun(
                idx=sample_indices,
                data=batched_data,
                return_a=True,
                return_b=False
            )
            fim_b = map_fun(
                idx=sample_indices,
                data=batched_data,
                return_a=False,
                return_b=True
            )
            fim_ab = map_fun(
                idx=sample_indices,
                data=batched_data,
                return_a=True,
                return_b=True
            )

        # Save as variables:
        # With relay across tf.Variable:
        self.fim_a = tf.Variable(tf.zeros([model_vars.n_features,
                                           model_vars.a_var.shape[0],
                                           model_vars.a_var.shape[0]], dtype=dtype), dtype=dtype)
        self.fim_b = tf.Variable(tf.zeros([model_vars.n_features,
                                           model_vars.b_var.shape[0],
                                           model_vars.b_var.shape[0]], dtype=dtype), dtype=dtype)
        self.fim_ab = (self.fim_a, self.fim_b)

        self.fim_a_set = tf.assign(self.fim_a, fim_a[0])
        self.fim_b_set = tf.assign(self.fim_b, fim_b[1])
        self.fim_ab_set = tf.group(tf.assign(self.fim_a, fim_ab[0]),
                                   tf.assign(self.fim_b, fim_ab[1]))

        # Without relay across tf.Variable:
        #self.fim_a = fim_ab[0]
        #self.fim_b = fim_ab[1]

    @abc.abstractmethod
    def analytic(
            self,
            sample_indices,
            batched_data,
            return_a,
            return_b
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