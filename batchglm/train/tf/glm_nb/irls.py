import tensorflow as tf

import logging

from .model import BasicModelGraph, ModelVars

from .external import op_utils
from .external import pkg_constants

logger = logging.getLogger(__name__)


class IRLS:
    """
    Compute the iteratively re-weighted least squares (IWLS or IRLS)
    parameter updates for a negative binomial GLM.
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
        if constraints_loc is not None and mode != "tf":
            raise ValueError("iwls does not work if constraints_loc is not None")
        if constraints_scale is not None and mode != "tf":
            raise ValueError("iwls does not work if constraints_scale is not None")

        self.noise_model = noise_model
        self._update_a = update_a
        self._update_b = update_b

        FIM, score = self.iwls_update(
            batched_data=batched_data,
            sample_indices=sample_indices,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            model_vars=model_vars,
            iterator=iterator,
            dtype=dtype
        )

        self.fim = FIM
        self.score = score

    def _constants_a(
            self,
            X,
            mu,
            r
    ):
        """
        Compute for mean model IWLS update for a negative binomial GLM.

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
        const = tf.divide(r, r+mu)
        W1 = tf.multiply(mu, const)
        W2 = tf.multiply(const, X - mu)

        return W1, W2

    def _constants_b(
            self,
            X,
            mu,
            r,
            log_r
    ):
        """
        Compute for dispersion model IWLS update for a negative binomial GLM.

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
        scalar_one = tf.constant(1, shape=(), dtype=X.dtype)
        scalar_two = tf.constant(2, shape=(), dtype=X.dtype)

        r_plus_mu = r+mu
        digamma_r = tf.math.digamma(x=r)
        digamma_r_plus_mu = tf.math.digamma(x=r_plus_mu)

        const1 = tf.multiply(scalar_two, tf.add(  # [observations, features]
            digamma_r,
            digamma_r_plus_mu
        ))
        const2 = tf.multiply(r, tf.add(  # [observations, features]
            tf.math.polygamma(a=scalar_one, x=r),
            tf.math.polygamma(a=scalar_one, x=r_plus_mu)
        ))
        const3 = tf.divide(r, r_plus_mu)
        W1 = tf.multiply(r, tf.add_n([const1, const2, const3]))

        W2 = tf.multiply(r, tf.add_n([  # [observations, features]
            digamma_r,
            digamma_r_plus_mu,
            tf.divide(r+X, r_plus_mu),
            log_r,
            scalar_one,
            tf.negative(tf.log(r_plus_mu))
        ]))

        return W1, W2

    def iwls_update(
            self,
            batched_data,
            sample_indices,
            constraints_loc,
            constraints_scale,
            model_vars: ModelVars,
            iterator,
            dtype
    ):
        """
        Compute the closed-form of the base_glm_all model hessian
        by evaluating its terms grouped by observations.

        Has three sub-functions which built the specific blocks of the hessian
        and one sub-function which concatenates the blocks into a full hessian.
        """

        def _a_byobs(X, design_loc, design_scale, mu, r, log_r):
            """
            Compute the mean model diagonal block of the
            closed form hessian of base_glm_all model by observation across features
            for a batch of observations.

            :param X: tf.tensor observations x features
                Observation by observation and feature.
            :param mu: tf.tensor observations x features
                Value of mean model by observation and feature.
            :param r: tf.tensor observations x features
                Value of dispersion model by observation and feature.
            """
            W1, W2 = self._constants_a(  # [observations x features]
                X=X,
                mu=mu,
                r=r,
                design_loc=design_loc
            )
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the einsum to efficiently perform the two outer products and the marginalisation.
            FIM = tf.einsum('ofc,od->fcd',
                            tf.einsum('of,oc->ofc', W1, design_loc),
                            design_loc)
            score = tf.matmul(tf.transpose(design_loc), W2)
            return FIM, score

        def _b_byobs(X, design_loc, design_scale, mu, r, log_r):
            """
            Compute the dispersion model diagonal block of the
            closed form hessian of base_glm_all model by observation across features.
            """
            W1, W2 = self._constants_b(  # [observations=1 x features]
                X=X,
                mu=mu,
                r=r,
                log_r=log_r
            )
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the Einstein summation to efficiently perform the two outer products and the marginalisation.
            FIM = tf.einsum('ofc,od->fcd',
                            tf.einsum('of,oc->ofc', W1, design_loc),
                            design_loc)
            score = tf.matmul(tf.transpose(design_loc), W2)
            return FIM, score

        def _assemble_byobs(idx, data):
            """
            Assemble hessian of a single observation across all features.

            This function runs the data batch (an observation) through the
            model graph and calls the wrappers that compute the
            individual closed forms of the hessian.

            :param data: tuple
                Containing the following parameters:
                - X: tf.tensor observations x features
                    Observation by observation and feature.
                - size_factors: tf.tensor observations x features
                    Model size factors by observation and feature.
                - params: tf.tensor features x coefficients
                    Estimated model variables.
            :return H: tf.tensor features x coefficients x coefficients
                Hessian evaluated on a single observation, provided in data.
            """
            X, design_loc, design_scale, size_factors = data
            a_split, b_split = tf.split(params, tf.TensorShape([p_shape_a, p_shape_b]))

            model = BasicModelGraph(
                X=X,
                design_loc=design_loc,
                design_scale=design_scale,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                a=a_split,
                b=b_split,
                dtype=dtype,
                size_factors=size_factors
            )
            mu = model.mu
            r = model.r
            log_r = tf.log(model.r)  # TODO inefficient

            if self._update_a and self._update_b:
                FIM_a, score_a = _a_byobs(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r, log_r=log_r)
                FIM_b, score_b = _b_byobs(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r, log_r=log_r)
                FIM = tf.concat([FIM_a, FIM_b], axis=1)
                score = tf.concat([score_a, score_b], axis=1)
            elif self._update_a and not self._update_b:
                FIM, score = _a_byobs(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r, log_r=log_r)
            elif not self._update_a and self._update_b:
                FIM, score = _b_byobs(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r, log_r=log_r)
            else:
                    raise ValueError("either require hess_a or hess_b")

            return FIM, score

        def _red(prev, cur):
            """
            Reduction operation for fisher information matrix computation across observations.

            Every evaluation of the hessian on an observation yields a full
            hessian matrix. This function sums over consecutive evaluations
            of this hessian so that not all separate evaluations have to be
            stored.
            """
            return tf.add(prev[0], cur[0]), tf.add(prev[1], cur[1]), tf.add(prev[2], cur[2]), tf.add(prev[3], cur[3])

        params = model_vars.params
        p_shape_a = model_vars.a_var.shape[0]  # This has to be _var to work with constraints.
        p_shape_b = model_vars.b_var.shape[0]  # This has to be _var to work with constraints.

        if iterator:
            FIM, score = op_utils.map_reduce(
                last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
                data=batched_data,
                map_fn=_assemble_byobs,
                reduce_fn=_red,
                parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
            )
        else:
            FIM, score = _assemble_byobs(
                idx=sample_indices,
                data=batched_data
            )

        return FIM, score
