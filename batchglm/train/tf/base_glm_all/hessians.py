import logging
from typing import List

import numpy as np
import tensorflow as tf

from .external import op_utils
from .external import pkg_constants
from .external import ModelVarsGLM

logger = logging.getLogger(__name__)


class HessianTF:
    """
    Compute the model hessian by gene using gradients from tensorflow.
    """

    noise_model: str

    _compute_hess_a: bool
    _compute_hess_b: bool

    def tf_byfeature(
            self,
            batched_data,
            sample_indices,
            constraints_loc: np.ndarray,
            constraints_scale: np.ndarray,
            model_vars: ModelVarsGLM,
            iterator,
            dtype
    ) -> List[tf.Tensor]:
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

        def feature_wises_batch(
                X,
                design_loc,
                design_scale,
                constraints_loc,
                constraints_scale,
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
            if self.noise_model == "nb":
                from .external_nb import BasicModelGraph
            else:
                raise ValueError("noise model %s was not recognized" % self.noise_model)

            dtype = X.dtype

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

                a_split, b_split = tf.split(params, tf.TensorShape([p_shape_a, p_shape_b]))

                # Define the model graph based on which the likelihood is evaluated
                # which which the hessian is computed:
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

        def _map(idx, data):
            X, design_loc, design_scale, size_factors = data
            return feature_wises_batch(
                X=X,
                design_loc=design_loc,
                design_scale=design_scale,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                params=model_vars.params,
                p_shape_a=model_vars.a_var.shape[0],  # This has to be _var to work with constraints.
                p_shape_b=model_vars.b_var.shape[0],  # This has to be _var to work with constraints.
                dtype=dtype,
                size_factors=size_factors
            )

        def _red(prev, cur):
            return [tf.add(p, c) for p, c in zip(prev, cur)]

        if iterator:
            H = op_utils.map_reduce(
                last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
                data=batched_data,
                map_fn=_map,
                reduce_fn=_red,
                parallel_iterations=1
            )
        else:
            H = _map(
                idx=sample_indices,
                data=batched_data
            )

        return H[0]