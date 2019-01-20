import logging
from typing import List, Tuple

import tensorflow as tf

from .external import pkg_constants
from .external import HessiansGLM

logger = logging.getLogger(__name__)


class HessianGLMALL(HessiansGLM):
    """
    Compute the Hessian matrix for a GLM by gene using gradients from tensorflow.
    """

    def byobs(
            self,
            sample_indices,
            batched_data
    ) -> Tuple:
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

        def _aa_byobs_batched(X, design_loc, mu, r):
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
                X=X,
                mu=mu,
                r=r,
            )
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the einsum to efficiently perform the two outer products and the marginalisation.
            XH = tf.matmul(design_loc, self.constraints_loc)
            Hblock = tf.einsum('ofc,od->fcd',
                               tf.einsum('of,oc->ofc', W, XH),
                               XH)
            return Hblock

        def _bb_byobs_batched(X, design_scale, mu, r):
            """
            Compute the dispersion model diagonal block of the
            closed form hessian of base_glm_all model by observation across features.
            """
            W = self._W_bb(  # [observations=1 x features]
                X=X,
                mu=mu,
                r=r,
            )
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the Einstein summation to efficiently perform the two outer products and the marginalisation.
            XH = tf.matmul(design_scale, self.constraints_scale)
            Hblock = tf.einsum('ofc,od->fcd',
                               tf.einsum('of,oc->ofc', W, XH),
                               XH)
            return Hblock

        def _ab_byobs_batched(X, design_loc, design_scale, mu, r):
            """
            Compute the mean-dispersion model off-diagonal block of the
            closed form hessian of base_glm_all model by observastion across features.

            Note that there are two blocks of the same size which can
            be compute from each other with a transpose operation as
            the hessian is symmetric.
            """
            W = self._W_ab(  # [observations=1 x features]
                X=X,
                mu=mu,
                r=r,
            )
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the Einstein summation to efficiently perform the two outer products and the marginalisation.
            XHloc = tf.matmul(design_loc, self.constraints_loc)
            XHscale = tf.matmul(design_scale, self.constraints_scale)
            Hblock = tf.einsum('ofc,od->fcd',
                               tf.einsum('of,oc->ofc', W, XHloc),
                               XHscale)
            return Hblock

        def assemble_batch(idx, data):
            """
            Assemble hessian of a batch of observations across all features.

            This function runs the data batch through the
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

            if self._compute_hess_a and self._compute_hess_b:
                H_aa = _aa_byobs_batched(
                    X=X,
                    design_loc=design_loc,
                    mu=mu,
                    r=r
                )
                H_bb = _bb_byobs_batched(
                    X=X,
                    design_scale=design_scale,
                    mu=mu,
                    r=r
                )
                H_ab = _ab_byobs_batched(
                    X=X,
                    design_loc=design_loc,
                    design_scale=design_scale,
                    mu=mu,
                    r=r
                )
                H_ba = tf.transpose(H_ab, perm=[0, 2, 1])
                H = tf.concat(
                    [tf.concat([H_aa, H_ab], axis=2),
                     tf.concat([H_ba, H_bb], axis=2)],
                    axis=1
                )
            elif self._compute_hess_a and not self._compute_hess_b:
                H = _aa_byobs_batched(
                    X=X,
                    design_loc=design_loc,
                    mu=mu,
                    r=r
                )
            elif not self._compute_hess_a and self._compute_hess_b:
                H = _bb_byobs_batched(
                    X=X,
                    design_scale=design_scale,
                    mu=mu,
                    r=r
                )
            else:
                raise ValueError("either require hess_a or hess_b")

            return H

        p_shape_a = self.model_vars.a_var.shape[0]  # This has to be _var to work with constraints.
        p_shape_b = self.model_vars.b_var.shape[0]  # This has to be _var to work with constraints.

        return assemble_batch(idx=sample_indices, data=batched_data)

    def byfeature(
            self,
            sample_indices,
            batched_data
    ) -> Tuple:
        """
        Compute the closed-form of the base_glm_all model hessian
        by evaluating its terms grouped by features.


        Has three sub-functions which built the specific blocks of the hessian
        and one sub-function which concatenates the blocks into a full hessian.
        """
        if self.noise_model == "nb":
            from .external_nb import BasicModelGraph
        else:
            raise ValueError("noise model %s was not recognized" % self.noise_model)

        def _aa_byfeature(
                X,
                design_loc,
                mu,
                r
        ):
            """
            Compute the mean model diagonal block of the
            closed form hessian of base_glm_all model by feature across observation.

            :param X: tf.tensor observations x features
                Observation by observation and feature.
            :param mu: tf.tensor observations x features
                Value of mean model by observation and feature.
            :param r: tf.tensor observations x features
                Value of dispersion model by observation and feature.
            """
            W = self._W_aa(  # [observations x features=1]
                X=X,
                mu=mu,
                r=r,
            )
            # The second dimension of const is only one element long,
            # this was a feature before but is no recycled into coefficients.
            # const = tf.broadcast_to(const, shape=design_loc.shape)  # [observations, coefficients]
            XH = tf.matmul(design_loc, self.constraints_loc)
            Hblock = tf.matmul(  # [coefficients, coefficients]
                tf.transpose(XH),  # [coefficients, observations]
                tf.multiply(XH, W)  # [observations, coefficients]
            )
            return Hblock

        def _bb_byfeature(
                X,
                design_scale,
                mu,
                r
        ):
            """
            Compute the dispersion model diagonal block of the
            closed form hessian of base_glm_all model by feature across observation.
            """
            W = self._W_bb(  # [observations x features=1]
                X=X,
                mu=mu,
                r=r,
            )
            # The second dimension of const is only one element long,
            # this was a feature before but is no recycled into coefficients.
            # const = tf.broadcast_to(const, shape=design_scale.shape)  # [observations, coefficients]
            XH = tf.matmul(design_scale, self.constraints_scale)
            Hblock = tf.matmul(  # [coefficients, coefficients]
                tf.transpose(XH),  # [coefficients, observations]
                tf.multiply(XH, W)  # [observations, coefficients]
            )
            return Hblock

        def _ab_byfeature(
                X,
                design_loc,
                design_scale,
                mu,
                r
        ):
            """
            Compute the mean-dispersion model off-diagonal block of the
            closed form hessian of base_glm_all model by feature across observation.

            Note that there are two blocks of the same size which can
            be compute from each other with a transpose operation as
            the hessian is symmetric.
            """
            W = self._W_ab(  # [observations x features=1]
                X=X,
                mu=mu,
                r=r,
            )
            # The second dimension of const is only one element long,
            # this was a feature before but is no recycled into coefficients_scale.
            # const = tf.broadcast_to(const, shape=design_scale.shape)  # [observations, coefficients_scale]
            XHloc = tf.matmul(design_loc, self.constraints_loc)
            XHscale = tf.matmul(design_scale, self.constraints_scale)
            Hblock = tf.matmul(  # [coefficients_loc, coefficients_scale]
                tf.transpose(XHloc),  # [coefficients_loc, observations]
                tf.multiply(XHscale, W)  # [observations, coefficients_scale]
            )
            return Hblock

        def assemble_batch(idx, data):
            def _assemble_byfeature(data):
                """
                Assemble hessian of a single feature.

                :param data: tuple
                    Containing the following parameters:
                    - X_t: tf.tensor observations x features .T
                        Observation by observation and feature.
                    - size_factors_t: tf.tensor observations x features .T
                        Model size factors by observation and feature.
                    - params_t: tf.tensor features x coefficients .T
                        Estimated model variables.
                """
                X_t, size_factors_t, params_t = data
                X = tf.transpose(X_t)
                size_factors = tf.transpose(size_factors_t)
                params = tf.transpose(params_t)  # design_params x features
                a_split, b_split = tf.split(params, tf.TensorShape([p_shape_a, p_shape_b]))

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

                if self._compute_hess_a and self._compute_hess_b:
                    H_aa = _aa_byfeature(
                        X=X,
                        design_loc=design_loc,
                        constraints_loc=constraints_loc,
                        mu=mu,
                        r=r
                    )
                    H_bb = _bb_byfeature(
                        X=X,
                        design_scale=design_scale,
                        mu=mu,
                        r=r
                    )
                    H_ab = _ab_byfeature(
                        X=X,
                        design_loc=design_loc,
                        design_scale=design_scale,
                        mu=mu,
                        r=r
                    )
                    H_ba = tf.transpose(H_ab, perm=[1, 0])
                    H = tf.concat(
                        [tf.concat([H_aa, H_ab], axis=1),
                         tf.concat([H_ba, H_bb], axis=1)],
                        axis=0
                    )
                elif self._compute_hess_a and self._compute_hess_b:
                    H = _aa_byfeature(
                        X=X,
                        design_loc=design_loc,
                        mu=mu,
                        r=r
                    )
                elif self._compute_hess_a and self._compute_hess_b:
                    H = _bb_byfeature(
                        X=X,
                        design_scale=design_scale,
                        mu=mu,
                        r=r
                    )
                else:
                    raise ValueError("either require hess_a or hess_b")

                return [H]

            X, design_loc, design_scale, size_factors = data
            X_t = tf.transpose(tf.expand_dims(X, axis=0), perm=[2, 0, 1])
            size_factors_t = tf.transpose(tf.expand_dims(size_factors, axis=0), perm=[2, 0, 1])
            params_t = tf.transpose(tf.expand_dims(self.model_vars.params, axis=0), perm=[2, 0, 1])

            H = tf.map_fn(
                fn=_assemble_byfeature,
                elems=(X_t, size_factors_t, params_t),
                dtype=[dtype],
                parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
            )

            return H

        p_shape_a = self.model_vars.a_var.shape[0]  # This has to be _var to work with constraints.
        p_shape_b = self.model_vars.b_var.shape[0]  # This has to be _var to work with constraints.

        return assemble_batch(idx=sample_indices, data=batched_data)

    def tf_byfeature(
            self,
            sample_indices,
            batched_data
    ) -> Tuple:
        """
        Compute hessians via tf.hessian for all gene-wise models separately.

        Contains three functions:

            - feature_wises_batch():
            a function that computes all hessians for a given batch
            of data by distributing the computation across features.
            - hessian_map():
            a function that unpacks the data from the iterator to run
            feature_wises_batch.
            - hessian_red():
            a function that performs the reduction of the hessians across hessians
            into a single hessian during the iteration over batches.
        """
        if self.noise_model == "nb":
            from .external_nb import BasicModelGraph
        else:
            raise ValueError("noise model %s was not recognized" % self.noise_model)

        def feature_wises_batch(
                X,
                design_loc,
                design_scale,
                params,
                p_shape_a,
                p_shape_b,
                dtype,
                size_factors=None
        ) -> List[tf.Tensor]:
            """
            Compute hessians via tf.hessian for all gene-wise models separately
            for a given batch of data.
            """
            # Hessian computation will be mapped across genes/features.
            # The map function maps across dimension zero, the slices have to
            # be 2D tensors to fit into BasicModelGraph, accordingly,
            # X, size_factors and params have to be reshaped to have genes in the first dimension
            # and cells or parameters with an extra padding dimension in the second
            # and third dimension. Note that size_factors is not a 1xobservations array
            # but is implicitly broadcasted to observations x features in Estimator.
            X_t = tf.transpose(tf.expand_dims(X, axis=0), perm=[2, 0, 1])
            size_factors_t = tf.transpose(tf.expand_dims(size_factors, axis=0), perm=[2, 0, 1])
            params_t = tf.transpose(tf.expand_dims(params, axis=0), perm=[2, 0, 1])

            def hessian(data):
                """ Helper function that computes hessian for a given gene.

                :param data: tuple (X_t, size_factors_t, params_t)
                """
                # Extract input data:
                X_t, size_factors_t, params_t = data
                size_factors = tf.transpose(size_factors_t)  # observations x features
                X = tf.transpose(X_t)  # observations x features
                params = tf.transpose(params_t)  # design_params x features

                a_split, b_split = tf.split(self.model_vars, params, tf.TensorShape([p_shape_a, p_shape_b]))

                # Define the model graph based on which the likelihood is evaluated
                # which which the hessian is computed:
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

                # Compute the hessian of the model of the given gene:
                if self._compute_hess_a and self._compute_hess_b:
                    H = tf.hessians(model.log_likelihood, params)
                elif self._compute_hess_a and not self._compute_hess_b:
                    H = tf.hessians(model.log_likelihood, a_split)
                elif not self._compute_hess_a and self._compute_hess_b:
                    H = tf.hessians(model.log_likelihood, b_split)
                else:
                    raise ValueError("either require hess_a or hess_b")

                return H

            # Map hessian computation across genes
            H = tf.map_fn(
                fn=hessian,
                elems=(X_t, size_factors_t, params_t),
                dtype=[dtype],
                parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
            )

            H = [tf.squeeze(tf.squeeze(tf.stack(h), axis=2), axis=3) for h in H]

            return H

        def assemble_batch(idx, data):
            X, design_loc, design_scale, size_factors = data
            return feature_wises_batch(
                X=X,
                design_loc=design_loc,
                design_scale=design_scale,
                params=self.model_vars.params,
                p_shape_a=self.model_vars.a_var.shape[0],  # This has to be _var to work with constraints.
                p_shape_b=self.model_vars.b_var.shape[0],  # This has to be _var to work with constraints.
                dtype=self.dtype,
                size_factors=size_factors
            )

        return assemble_batch(idx=sample_indices, data=batched_data)