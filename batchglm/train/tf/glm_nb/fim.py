import tensorflow as tf

import logging

from .external import FIMGLM
from .model import BasicModelGraph, ModelVars

from .external import op_utils
from .external import pkg_constants

logger = logging.getLogger(__name__)


class FIM(FIMGLM):
    """
    Compute the iteratively re-weighted least squares (IWLS or IRLS)
    parameter updates for a negative binomial GLM.
    """
    def _constants_a(
            self,
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
        W = tf.multiply(mu, const)

        return W

    def _constants_b(
            self,
            X,
            mu,
            r
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
        W = tf.multiply(r, tf.add_n([const1, const2, const3]))

        return W

    def analytic(
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

        def _a_byobs(design_loc, constraints_loc, mu, r):
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
            W = self._constants_a(  # [observations x features]
                mu=mu,
                r=r
            )
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the einsum to efficiently perform the two outer products and the marginalisation.
            XH = tf.matmul(design_loc, constraints_loc)
            FIM = tf.einsum('ofc,od->fcd',
                            tf.einsum('of,oc->ofc', W, XH),
                            XH)
            return FIM

        def _b_byobs(X, design_scale, constraints_scale, mu, r):
            """
            Compute the dispersion model diagonal block of the
            closed form hessian of base_glm_all model by observation across features.
            """
            W = self._constants_b(  # [observations=1 x features]
                X=X,
                mu=mu,
                r=r
            )
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the Einstein summation to efficiently perform the two outer products and the marginalisation.
            XH = tf.matmul(design_scale, constraints_scale)
            FIM = tf.einsum('ofc,od->fcd',
                            tf.einsum('of,oc->ofc', W, XH),
                            XH)
            return FIM

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

            # The full fisher information matrix is block-diagonal with the cross-model
            # blocks all zero. Accordingly, mean and dispersion model updates can be
            # treated independently and the full fisher information matrix is never required.
            # Here, the non-zero model-wise diagonal blocks are computed and returned
            # as a dictionary. The according score function vectors are also returned as a dictionary.
            if self._update_a and self._update_b:
                fim_a = _a_byobs(design_loc=design_loc, constraints_loc=constraints_loc, mu=mu, r=r)
                fim_b = _b_byobs(X=X, design_scale=design_scale, constraints_scale=constraints_scale, mu=mu, r=r)
            elif self._update_a and not self._update_b:
                fim_a = _a_byobs(design_loc=design_loc, constraints_loc=constraints_loc, mu=mu, r=r)
                fim_b = tf.zeros(shape=())
            elif not self._update_a and self._update_b:
                fim_a = tf.zeros(shape=())
                fim_b = _b_byobs(X=X, design_scale=design_scale, constraints_scale=constraints_scale, mu=mu, r=r)
            else:
                raise ValueError("either require hess_a or hess_b")

            return fim_a, fim_b

        def _red(prev, cur):
            """
            Reduction operation for fisher information matrix computation across observations.

            Every evaluation of the hessian on an observation yields a full
            hessian matrix. This function sums over consecutive evaluations
            of this hessian so that not all separate evaluations have to be
            stored.
            """
            fim_a = tf.add(prev[0], cur[0])
            fim_b = tf.add(prev[1], cur[1])
            return fim_a, fim_b

        params = model_vars.params
        p_shape_a = model_vars.a_var.shape[0]  # This has to be _var to work with constraints.
        p_shape_b = model_vars.b_var.shape[0]  # This has to be _var to work with constraints.

        if iterator:
            fims = op_utils.map_reduce(
                last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
                data=batched_data,
                map_fn=_assemble_byobs,
                reduce_fn=_red,
                parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
            )
        else:
            fims = _assemble_byobs(
                idx=sample_indices,
                data=batched_data
            )

        return fims
