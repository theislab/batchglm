import tensorflow as tf

import logging

from .external import FIMGLM

logger = logging.getLogger(__name__)


class FIMGLMALL(FIMGLM):
    """
    Compute the iteratively re-weighted least squares (IWLS or IRLS)
    parameter updates for a negative binomial GLM.
    """

    def analytic(
            self,
            sample_indices,
            batched_data,
            return_a,
            return_b
    ):
        """
        Compute the closed-form of the base_glm_all model hessian
        by evaluating its terms grouped by observations.

        Has three sub-functions which built the specific blocks of the hessian
        and one sub-function which concatenates the blocks into a full hessian.
        """
        if self.noise_model == "nb":
            from .external_nb import BasicModelGraph
        else:
            raise ValueError("noise model %s was not recognized" % self.noise_model)

        def _a_byobs(design_loc, mu, r):
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
            W = self._W_aa(  # [observations x features]
                mu=mu,
                r=r
            )
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the einsum to efficiently perform the two outer products and the marginalisation.
            if self.constraints_loc is not None:
                XH = tf.matmul(design_loc, self.constraints_loc)
            else:
                XH = design_loc

            fim = tf.einsum('ofc,od->fcd',
                            tf.einsum('of,oc->ofc', W, XH),
                            XH)
            return fim

        def _b_byobs(X, design_scale, mu, r):
            """
            Compute the dispersion model diagonal block of the
            closed form hessian of base_glm_all model by observation across features.
            """
            W = self._W_bb(  # [observations=1 x features]
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
            if self.constraints_scale is not None:
                XH = tf.matmul(design_scale, self.constraints_scale)
            else:
                XH = design_scale

            fim = tf.einsum('ofc,od->fcd',
                            tf.einsum('of,oc->ofc', W, XH),
                            XH)
            return fim

        def assemble_batch(idx, data, return_a, return_b):
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
            a_split, b_split = tf.split(self.model_vars.params, tf.TensorShape([p_shape_a, p_shape_b]))

            model = BasicModelGraph(
                X=X,
                design_loc=design_loc,
                design_scale=design_scale,
                constraints_loc=self.constraints_loc,
                constraints_scale=self.constraints_scale,
                a_var=a_split,
                b_var=b_split,
                dtype=self.dtype,
                size_factors=size_factors
            )
            mu = model.mu
            r = model.r

            # The full fisher information matrix is block-diagonal with the cross-model
            # blocks all zero. Accordingly, mean and dispersion model updates can be
            # treated independently and the full fisher information matrix is never required.
            # Here, the non-zero model-wise diagonal blocks are computed and returned
            # as a dictionary. The according score function vectors are also returned as a dictionary.
            if return_a and return_b:
                fim_a = _a_byobs(design_loc=design_loc, mu=mu, r=r)
                fim_b = _b_byobs(X=X, design_scale=design_scale, mu=mu, r=r)
            elif return_a and not return_b:
                fim_a = _a_byobs(design_loc=design_loc, mu=mu, r=r)
                fim_b = tf.zeros((), dtype=self.dtype)
            elif not return_a and return_b:
                fim_a = tf.zeros((), dtype=self.dtype)
                fim_b = _b_byobs(X=X, design_scale=design_scale, mu=mu, r=r)
            else:
                assert False

            fim = (fim_a, fim_b)
            return fim

        p_shape_a = self.model_vars.a_var.shape[0]  # This has to be _var to work with constraints.
        p_shape_b = self.model_vars.b_var.shape[0]  # This has to be _var to work with constraints.

        fim = assemble_batch(idx=sample_indices, data=batched_data, return_a=return_a, return_b=return_b)
        return fim
