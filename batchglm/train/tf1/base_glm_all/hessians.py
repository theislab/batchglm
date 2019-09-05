import logging

import tensorflow as tf

from .external import pkg_constants
from .external import HessiansGLM

logger = logging.getLogger(__name__)


class HessianGLMALL(HessiansGLM):
    """
    Compute the Hessian matrix for a GLM by gene using gradients from tensorflow.
    """

    def hessian_analytic(
            self,
            model
    ) -> tf.Tensor:
        """
        Compute the closed-form of the base_glm_all model hessian
        by evaluating its terms grouped by observations.

        Has three sub-functions which built the specific blocks of the hessian
        and one sub-function which concatenates the blocks into a full hessian.
        """

        def _aa_byobs_batched(model):
            """
            Compute the mean model diagonal block of the
            closed form hessian of base_glm_all model by observation across features
            for a batch of observations.
            """
            W = self._weight_hessian_aa(  # [observations x features]
                X=model.X,
                loc=model.model_loc,
                scale=model.model_scale,
            )
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the einsum to efficiently perform the two outer products and the marginalisation.
            if self.constraints_loc is not None:
                XH = tf.matmul(model.design_loc, model.constraints_loc)
            else:
                XH = model.design_loc

            Hblock = tf.einsum('ofc,od->fcd',
                               tf.einsum('of,oc->ofc', W, XH),
                               XH)
            return Hblock

        def _bb_byobs_batched(model):
            """
            Compute the dispersion model diagonal block of the
            closed form hessian of base_glm_all model by observation across features.
            """
            W = self._weight_hessian_bb(  # [observations=1 x features]
                X=model.X,
                loc=model.model_loc,
                scale=model.model_scale,
            )
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the Einstein summation to efficiently perform the two outer products and the marginalisation.
            if self.constraints_scale is not None:
                XH = tf.matmul(model.design_scale, model.constraints_scale)
            else:
                XH = model.design_scale

            Hblock = tf.einsum('ofc,od->fcd',
                               tf.einsum('of,oc->ofc', W, XH),
                               XH)
            return Hblock

        def _ab_byobs_batched(model):
            """
            Compute the mean-dispersion model off-diagonal block of the
            closed form hessian of base_glm_all model by observastion across features.

            Note that there are two blocks of the same size which can
            be compute from each other with a transpose operation as
            the hessian is symmetric.
            """
            W = self._weight_hessian_ab(  # [observations=1 x features]
                X=model.X,
                loc=model.model_loc,
                scale=model.model_scale,
            )
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the Einstein summation to efficiently perform the two outer products and the marginalisation.
            if self.constraints_loc is not None:
                XHloc = tf.matmul(model.design_loc, model.constraints_loc)
            else:
                XHloc = model.design_loc

            if self.constraints_scale is not None:
                XHscale = tf.matmul(model.design_scale, model.constraints_scale)
            else:
                XHscale = model.design_scale

            Hblock = tf.einsum('ofc,od->fcd',
                               tf.einsum('of,oc->ofc', W, XHloc),
                               XHscale)
            return Hblock

        if self.compute_a and self.compute_b:
            H_aa = _aa_byobs_batched(model=model)
            H_bb = _bb_byobs_batched(model=model)
            H_ab = _ab_byobs_batched(model=model)
            H_ba = tf.transpose(H_ab, perm=[0, 2, 1])
            H = tf.concat(
                [tf.concat([H_aa, H_ab], axis=2),
                 tf.concat([H_ba, H_bb], axis=2)],
                axis=1
            )
        elif self.compute_a and not self.compute_b:
            H = _aa_byobs_batched(model=model)
        elif not self.compute_a and self.compute_b:
            H = _bb_byobs_batched(model=model)
        else:
            H = tf.zeros((), dtype=self.dtype)

        return H

    def hessian_tf(
            self,
            model
    ) -> tf.Tensor:
        """
        Compute hessians via tf1.gradients for all gene-wise in parallel.
        """
        if self.compute_a and self.compute_b:
            var_shape = tf.shape(self.model_vars.params)
            var = self.model_vars.params
        elif self.compute_a and not self.compute_b:
            var_shape = tf.shape(self.model_vars.a_var)
            var = self.model_vars.a_var
        elif not self.compute_a and self.compute_b:
            var_shape = tf.shape(self.model_vars.b_var)
            var = self.model_vars.b_var

        if self.compute_a or self.compute_b:
            # Compute first order derivatives as first step to get second order derivatives.
            first_der = tf.gradients(model.log_likelihood, var)[0]

            # Note on error comment below: The arguments that cause the error, infer_shape and element_shape,
            # are not necessary for this code but would provide an extra layer of stability as all
            # elements of the array have the same shape.
            loop_vars = [
                tf.constant(0, tf.int32),  # iteration counter
                tf.TensorArray(  # hessian slices [:,:,j]
                    dtype=var.dtype,
                    size=var_shape[0],
                    clear_after_read=False
                    #infer_shape=True,  # TODO tf1>=2.0: this causes error related to eager execution in tf1.12
                    #element_shape=var_shape
                )
            ]

            # Compute second order derivatives based on parameter-wise slices of the tensor of first order derivatives.
            _, h_tensor_array = tf.while_loop(
                cond=lambda i, _: i < var_shape[0],
                body=lambda i, result: (
                    i + 1,
                    result.write(
                        index=i,
                        value=tf.gradients(first_der[i, :], var)[0]
                    )
                ),
                loop_vars=loop_vars,
                return_same_structure=True
            )

            # h_tensor_array is a TensorArray, reshape this into a tensor so that it can be used
            # in down-stream computation graphs.
            h = tf.transpose(tf.reshape(
                h_tensor_array.stack(),
                tf.stack((var_shape[0], var_shape[0], var_shape[1]))
            ), perm=[2, 1, 0])
        else:
            h = tf.zeros((), dtype=self.dtype)

        return h