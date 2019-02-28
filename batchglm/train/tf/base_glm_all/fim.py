import tensorflow as tf

import logging

from .external import FIMGLM

logger = logging.getLogger(__name__)


class FIMGLMALL(FIMGLM):
    """
    Compute the iteratively re-weighted least squares (IWLS or IRLS)
    parameter updates for a negative binomial GLM.
    """

    def fim_analytic(
            self,
            model
    ):
        """
        Compute the closed-form of the base_glm_all model hessian
        by evaluating its terms grouped by observations.

        Has three sub-functions which built the specific blocks of the hessian
        and one sub-function which concatenates the blocks into a full hessian.
        """

        def _a_byobs(model):
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
            W = self._weight_fim_aa(  # [observations x features]
                mu=model.mu,
                r=model.r
            )
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the einsum to efficiently perform the two outer products and the marginalisation.
            if self.constraints_loc is not None:
                XH = tf.matmul(model.design_loc, self.constraints_loc)
            else:
                XH = model.design_loc

            fim = tf.einsum('ofc,od->fcd',
                            tf.einsum('of,oc->ofc', W, XH),
                            XH)
            return fim

        def _b_byobs(model):
            """
            Compute the dispersion model diagonal block of the
            closed form hessian of base_glm_all model by observation across features.
            """
            W = self._weight_fim_bb(  # [observations=1 x features]
                X=model.X,
                mu=model.mu,
                r=model.r
            )
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the Einstein summation to efficiently perform the two outer products and the marginalisation.
            if self.constraints_scale is not None:
                XH = tf.matmul(model.design_scale, self.constraints_scale)
            else:
                XH = model.design_scale

            fim = tf.einsum('ofc,od->fcd',
                            tf.einsum('of,oc->ofc', W, XH),
                            XH)
            return fim

        # The full fisher information matrix is block-diagonal with the cross-model
        # blocks all zero. Accordingly, mean and dispersion model updates can be
        # treated independently and the full fisher information matrix is never required.
        # Here, the non-zero model-wise diagonal blocks are computed and returned
        # as a dictionary. The according score function vectors are also returned as a dictionary.
        if self.compute_fim_a and self.compute_fim_b:
            fim_a = _a_byobs(model=model)
            fim_b = _b_byobs(model=model)
        elif self.compute_fim_a and not self.compute_fim_b:
            fim_a = _a_byobs(model=model)
            fim_b = tf.zeros((), dtype=self.dtype)
        elif not self.compute_fim_a and self.compute_fim_b:
            fim_a = tf.zeros((), dtype=self.dtype)
            fim_b = _b_byobs(model=model)
        else:
            fim_a = tf.zeros((), dtype=self.dtype)
            fim_b = tf.zeros((), dtype=self.dtype)

        return fim_a, fim_b
