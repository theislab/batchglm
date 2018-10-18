import abc
from typing import Union, Dict, Tuple, List
import logging
import pprint
from enum import Enum

import tensorflow as tf
# import tensorflow_probability as tfp

import numpy as np
from numpy.linalg import matrix_rank

try:
    import anndata
except ImportError:
    anndata = None

from .external import AbstractEstimator, XArrayEstimatorStore, InputData, Model, MonitoredTFEstimator, TFEstimatorGraph
from .external import nb_utils, train_utils, op_utils, rand_utils
from .external import pkg_constants
from .hessians import *

ESTIMATOR_PARAMS = AbstractEstimator.param_shapes().copy()
ESTIMATOR_PARAMS.update({
    "batch_probs": ("batch_observations", "features"),
    "batch_log_probs": ("batch_observations", "features"),
    "batch_log_likelihood": (),
    "full_loss": (),
    "full_gradient": ("features",),
})

logger = logging.getLogger(__name__)


def _hessian_nb_glm_aa_coef_invariant(
    X,
    mu,
    r,
    dtype
):
    """
    Compute the coefficient index invariant part of the
    mean model block of the hessian of nb_glm model.

    Below, X are design matrices of the mean (m)
    and dispersion (r) model respectively, Y are the
    observed data. Const is constant across all combinations
    of i and j.
    .. math::

        &H^{m,m}_{i,j} = -X^m_i*X^m_j*mu*\frac{Y/r+1}{(1+mu/r)^2} \\
        &const = -mu*\frac{Y/r+1}{(1+mu/r)^2} \\
        &H^{m,m}_{i,j} = X^m_i*X^m_j*const \\

    :param X: tf.tensor observations x features
        Observation by observation and feature.
    :param mu: tf.tensor observations x features
        Value of mean model by observation and feature.
    :param r: tf.tensor observations x features
        Value of dispersion model by observation and feature.
    :param dtype: dtype
    :return const: tf.tensor observations x features
        Coefficient invariant terms of hessian of
        given observations and features.
    """
    scalar_one = tf.constant(1, shape=[1, 1], dtype=dtype)
    const = tf.negative(tf.multiply(
        mu,  # [observations x features]
        tf.divide(
            tf.divide(X, r) + scalar_one,
            tf.square(scalar_one + tf.divide(mu, r))
        )
    ))
    return const


def _hessian_nb_glm_bb_coef_invariant(
    X,
    mu,
    r,
    dtype
):
    """
    Compute the coefficient index invariant part of the
    dispersion model block of the hessian of nb_glm model.
    
    Below, X are design matrices of the mean (m)
    and dispersion (r) model respectively, Y are the
    observed data. Const is constant across all combinations
    of i and j.
    .. math::

        H^{r,r}_{i,j}&= X^r_i*X^r_j \\
            &*r*\bigg(psi_0(r+Y)+\psi_0(r)+r*psi_1(r+Y)+r*psi_1(r) \\
            &-\frac{mu}{(r+mu)^2}*(r+Y) \\
            &+\frac{mu-r}{r+mu}+\log(r)+1-\log(r+mu) \bigg) \\
        const = r*\bigg(psi_0(r+Y)+\psi_0(r)+r*psi_1(r+Y)+r*psi_1(r) \\
            &-\frac{mu}{(r+mu)^2}*(r+Y) \\
            &+\frac{mu-r}{r+mu}+\log(r)+1-\log(r+mu) \bigg) \\
        H^{r,r}_{i,j}&= X^r_i*X^r_j * const \\
    
    :param X: tf.tensor observations x features
        Observation by observation and feature.
    :param mu: tf.tensor observations x features
        Value of mean model by observation and feature.
    :param r: tf.tensor observations x features
        Value of dispersion model by observation and feature.
    :param dtype: dtype
    :return const: tf.tensor observations x features
        Coefficient invariant terms of hessian of
        given observations and features.
    """
    scalar_zero = tf.constant(0, shape=[1, 1], dtype=dtype)
    scalar_one = tf.constant(1, shape=[1, 1], dtype=dtype)
    const1 = tf.add(  # [observations, features]
        tf.math.polygamma(a=scalar_zero, x=tf.add(r, X)),
        tf.add(
            tf.math.polygamma(a=scalar_zero, x=r),
            tf.add(
                tf.multiply(r, tf.math.polygamma(a=scalar_one, x=tf.add(r, X))),
                tf.multiply(r, tf.math.polygamma(a=scalar_one, x=r))
            )
        )
    )
    const2 = tf.multiply(  # [observations, features]
        tf.divide(mu, tf.square(tf.add(r, mu))),
        tf.add(r, X)
    )
    const3 = tf.add(  # [observations, features]
        tf.divide(tf.subtract(mu, r), tf.add(mu, r)),
        tf.subtract(
            tf.add(tf.log(r), scalar_one),
            tf.log(tf.add(r, mu))
        )
    )
    const = tf.add(tf.add(const1, const2), const3)  # [observations, features]
    const = tf.multiply(r, const)
    return const


def _hessian_nb_glm_ab_coef_invariant(
    X,
    mu,
    r,
    dtype
):
    """
    Compute the coefficient index invariant part of the
    mean-dispersion model block of the hessian of nb_glm model.

    Note that there are two blocks of the same size which can
    be compute from each other with a transpose operation as
    the hessian is symmetric.
    
    Below, X are design matrices of the mean (m)
    and dispersion (r) model respectively, Y are the
    observed data. Const is constant across all combinations
    of i and j.
    .. math::

        &H^{m,r}_{i,j} = X^m_i*X^r_j*mu*\frac{Y-mu}{(1+mu/r)^2} \\
        &H^{r,m}_{i,j} = X^m_i*X^r_j*r*mu*\frac{Y-mu}{(mu+r)^2} \\
        &const = r*mu*\frac{Y-mu}{(mu+r)^2} \\
        &H^{m,r}_{i,j} = X^m_i*X^r_j*const \\
        &H^{r,m}_{i,j} = X^m_i*X^r_j*const \\
    
    :param X: tf.tensor observations x features
        Observation by observation and feature.
    :param mu: tf.tensor observations x features
        Value of mean model by observation and feature.
    :param r: tf.tensor observations x features
        Value of dispersion model by observation and feature.
    :param dtype: dtype
    :return const: tf.tensor observations x features
        Coefficient invariant terms of hessian of
        given observations and features.
    """
    const = tf.multiply(
        tf.multiply(mu, r),  # [observations, features]
        tf.divide(
            X - mu, # [observations, features]
            tf.square(tf.add(mu, r))
        )
    )
    return const


def _hessian_nb_glm_byobs(
        batched_data,
        sample_indices,
        constraints_loc,
        constraints_scale,
        model_vars,
        dtype
):
    """
    Compute the closed-form of the nb_glm model hessian 
    by evalutating its terms grouped by observations.

    Has three subfunctions which built the specific blocks of the hessian
    and one subfunction which concatenates the blocks into a full hessian.

    TODO: compute in obs batches by using matmul across 3rd dim
    """

    def _hessian_nb_glm_aa_byobs(X, design_loc, design_scale, mu, r):
        """
        Compute the mean model diagonal block of the 
        closed form hessian of nb_glm model by observation across features.

        :param X: tf.tensor observations x features
            Observation by observation and feature.
        :param mu: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param r: tf.tensor observations x features
            Value of dispersion model by observation and feature.
        """
        scalar_one = tf.constant(1, shape=[1, 1], dtype=dtype)
        const = _hessian_nb_glm_aa_coef_invariant(  # [observations=1 x features]
            X=X,
            mu=mu,
            r=r,
            dtype=dtype
        )
        nonconst = tf.matmul(tf.transpose(design_loc), design_loc)  # [coefficients, coefficients]
        nonconst = tf.expand_dims(nonconst, axis=-1)  # [coefficients, coefficients, observations=1]
        Hblock = tf.transpose(tf.tensordot(  # [features, coefficients, coefficients]
            a=nonconst,  # [coefficients, coefficients, observations=1]
            b=const,  # [observations=1 x features]
            axes=1  # collapse last dimension of a and first dimension of b
        ))
        return Hblock

    def _hessian_nb_glm_bb_byobs(X, design_loc, design_scale, mu, r):
        """
        Compute the dispersion model diagonal block of the 
        closed form hessian of nb_glm model by observation across features.

        :param X: tf.tensor observations x features
            Observation by observation and feature.
        :param mu: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param r: tf.tensor observations x features
            Value of dispersion model by observation and feature.
        """
        const = _hessian_nb_glm_bb_coef_invariant(  # [observations=1 x features]
            X=X,
            mu=mu,
            r=r,
            dtype=dtype
        )
        nonconst = tf.matmul(tf.transpose(design_scale), design_scale)  # [coefficients, coefficients]
        nonconst = tf.expand_dims(nonconst, axis=-1)  # [coefficients, coefficients, observations=1]
        Hblock = tf.transpose(tf.tensordot(  # [features, coefficients, coefficients]
            a=nonconst,  # [coefficients, coefficients, observations=1]
            b=const,  # [observations=1 x features]
            axes=1  # collapse last dimension of a and first dimension of b
        ))
        return Hblock

    def _hessian_nb_glm_ab_byobs(X, design_loc, design_scale, mu, r):
        """
        Compute the mean-dispersion model off-diagonal block of the 
        closed form hessian of nb_glm model by observastion across features.

        Note that there are two blocks of the same size which can
        be compute from each other with a transpose operation as
        the hessian is symmetric.
        
        :param X: tf.tensor observations x features
            Observation by observation and feature.
        :param mu: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param r: tf.tensor observations x features
            Value of dispersion model by observation and feature.
        """
        const = _hessian_nb_glm_ab_coef_invariant(  # [observations=1 x features]
            X=X,
            mu=mu,
            r=r,
            dtype=dtype
        )
        nonconst = tf.matmul(tf.transpose(design_loc), design_scale)  # [coefficient_loc, coefficients_scale]
        nonconst = tf.expand_dims(nonconst, axis=-1)  # [coefficient_loc, coefficients_scale, observations=1]
        Hblock = tf.transpose(tf.tensordot(  # [features, coefficient_loc, coefficients_scale]
            a=nonconst,  # [coefficient_loc, coefficients_scale, observations=1]
            b=const,  # [observations=1 x features]
            axes=1  # collapse last dimension of a and first dimension of b
        ))
        return Hblock

    def _hessian_nb_glm_assemble_byobs(idx, data):
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

        H_aa = _hessian_nb_glm_aa_byobs(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r)
        H_bb = _hessian_nb_glm_bb_byobs(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r)
        H_ab = _hessian_nb_glm_ab_byobs(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r)
        H_ba = tf.transpose(H_ab, perm=[0, 2, 1])
        H = tf.concat(
            [tf.concat([H_aa, H_ab], axis=1),
             tf.concat([H_ba, H_bb], axis=1)],
            axis=2
        )
        return H

    def _hessian_red(prev, cur):
        """
        Reduction operation for hessian computation across observations.

        Every evaluation of the hessian on an observation yields a full 
        hessian matrix. This function sums over consecutive evaluations
        of this hessian so that not all seperate evluations have to be
        stored.
        """
        return tf.add(prev, cur)

    params = model_vars.params
    p_shape_a = model_vars.a_var.shape[0]
    p_shape_b = model_vars.b_var.shape[0]

    H = op_utils.map_reduce(
        last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
        data=batched_data,
        map_fn=_hessian_nb_glm_assemble_byobs,
        reduce_fn=_hessian_red,
        parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
    )
    return H


def _hessian_nb_glm_byfeature(
        batched_data,
        sample_indices,
        constraints_loc,
        constraints_scale,
        model_vars,
        dtype
):
    """
    Compute the closed-form of the nb_glm model hessian 
    by evalutating its terms grouped by features.


    Has three subfunctions which built the specific blocks of the hessian
    and one subfunction which concatenates the blocks into a full hessian.
    """

    def _hessian_nb_glm_aa_byfeature(X, design_loc, design_scale, mu, r):
        """
        Compute the mean model diagonal block of the 
        closed form hessian of nb_glm model by feature across observation.

        :param X: tf.tensor observations x features
            Observation by observation and feature.
        :param mu: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param r: tf.tensor observations x features
            Value of dispersion model by observation and feature.
        :param dtype: dtype
        """
        scalar_one = tf.constant(1, shape=[1, 1], dtype=dtype)
        const = _hessian_nb_glm_aa_coef_invariant(  # [observations x features=1]
            X=X,
            mu=mu,
            r=r,
            dtype=dtype
        )
        # The second dimension of const is only one element long,
        # this was a feature before but is no recycled into coefficients.
        #const = tf.broadcast_to(const, shape=design_loc.shape)  # [observations, coefficients]
        Hblock = tf.matmul(  # [coefficients, coefficients]
            tf.transpose(design_loc),  # [coefficients, observations]
            tf.multiply(design_loc, const)  # [observations, coefficients]
        )
        return Hblock

    def _hessian_nb_glm_bb_byfeature(X, design_loc, design_scale, mu, r):
        """
        Compute the dispersion model diagonal block of the 
        closed form hessian of nb_glm model by feature across observation.
        
        :param X: tf.tensor observations x features
            Observation by observation and feature.
            Dispersion model design matrix entries by observation and coefficient.
        :param mu: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param r: tf.tensor observations x features
            Value of dispersion model by observation and feature.
        :param dtype: dtype
        """
        const = _hessian_nb_glm_bb_coef_invariant(  # [observations x features=1]
            X=X,
            mu=mu,
            r=r,
            dtype=dtype
        )
        # The second dimension of const is only one element long,
        # this was a feature before but is no recycled into coefficients.
        #const = tf.broadcast_to(const, shape=design_scale.shape)  # [observations, coefficients]
        Hblock = tf.matmul(  # [coefficients, coefficients]
            tf.transpose(design_scale),  # [coefficients, observations]
            tf.multiply(design_scale, const)  # [observations, coefficients]
        )
        return Hblock

    def _hessian_nb_glm_ab_byfeature(X, design_loc, design_scale, mu, r):
        """
        Compute the mean-dispersion model off-diagonal block of the
        closed form hessian of nb_glm model by feature across observation.

        Note that there are two blocks of the same size which can
        be compute from each other with a transpose operation as
        the hessian is symmetric.

        :param X: tf.tensor observations x features
            Observation by observation and feature.
        :param mu: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param r: tf.tensor observations x features
            Value of dispersion model by observation and feature.
        :param dtype: dtype
        """
        const = _hessian_nb_glm_ab_coef_invariant(  # [observations x features=1]
            X=X,
            mu=mu,
            r=r,
            dtype=dtype
        )
        # The second dimension of const is only one element long,
        # this was a feature before but is no recycled into coefficients_scale.
        #const = tf.broadcast_to(const, shape=design_scale.shape)  # [observations, coefficients_scale]
        Hblock = tf.matmul(  # [coefficients_loc, coefficients_scale]
            tf.transpose(design_loc),  # [coefficients_loc, observations]
            tf.multiply(design_scale, const)  # [observations, coefficients_scale]
        )
        return Hblock

    def _hessian_map(idx, data):
        def _hessian_nb_glm_assemble_byfeature(data):
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
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                a=a_split,
                b=b_split,
                dtype=dtype,
                size_factors=size_factors
            )
            mu = model.mu
            r = model.r

            H_aa = _hessian_nb_glm_aa_byfeature(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r)
            H_bb = _hessian_nb_glm_bb_byfeature(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r)
            H_ab = _hessian_nb_glm_ab_byfeature(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r)
            H_ba = tf.transpose(H_ab, perm=[1, 0])

            H = tf.concat(
                [tf.concat([H_aa, H_ab], axis=1),
                 tf.concat([H_ba, H_bb], axis=1)],
                axis=0
            )
            return [H]

        X, design_loc, design_scale, size_factors = data
        X_t = tf.transpose(tf.expand_dims(X, axis=0), perm=[2, 0, 1])
        size_factors_t = tf.transpose(tf.expand_dims(size_factors, axis=0), perm=[2, 0, 1])
        params_t = tf.transpose(tf.expand_dims(params, axis=0), perm=[2, 0, 1])

        H = tf.map_fn(
            fn=_hessian_nb_glm_assemble_byfeature,
            elems=(X_t, size_factors_t, params_t),
            dtype=[dtype],
            parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
        )

        return H

    def _hessian_red(prev, cur):
        return [tf.add(p, c) for p, c in zip(prev, cur)]

    params = model_vars.params
    p_shape_a = model_vars.a_var.shape[0]
    p_shape_b = model_vars.b_var.shape[0]

    H = op_utils.map_reduce(
        last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
        data=batched_data,
        map_fn=_hessian_map,
        reduce_fn=_hessian_red,
        parallel_iterations=1,
    )
    H = H[0]
    return H

def _tf_hessian_nb_glm_byfeature(
        batched_data,
        sample_indices,
        constraints_loc,
        constraints_scale,
        model_vars,
        dtype
) -> List[tf.Tensor]:
    """ 
    Compute hessians via tf.hessian for all gene-wise models separately.

    Contains three functions:

        - feature_wise_hessians_batch():
        a function that computes all hessians for a given batch
        of data by distributing the computation across features. 
        - hessian_map():
        a function that unpacks the data from the iterator to run
        feature_wise_hessians_batch.
        - hessian_red():
        a function that performs the reduction of the hessians across hessians
        into a single hessian during the iteration over batches.
    """

    def feature_wise_hessians_batch(
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
            H = tf.hessians(- model.log_likelihood, params)
            return H

        # Map hessian computation across genes
        H = tf.map_fn(
            fn=hessian,
            elems=(X_t, size_factors_t, params_t),
            dtype=[dtype],  # hessians of [a, b]
            parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
        )

        H = [tf.squeeze(tf.squeeze(tf.stack(h), axis=2), axis=3) for h in H]

        return H

    def _hessian_map(idx, data):
        X, design_loc, design_scale, size_factors = data
        return feature_wise_hessians_batch(
            X=X,
            design_loc=design_loc,
            design_scale=design_scale,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            params=model_vars.params,
            p_shape_a=model_vars.a_var.shape[0],
            p_shape_b=model_vars.b_var.shape[0],
            dtype=dtype,
            size_factors=size_factors
        )

    def _hessian_red(prev, cur):
        return [tf.add(p, c) for p, c in zip(prev, cur)]

    H = op_utils.map_reduce(
        last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
        data=batched_data,
        map_fn=_hessian_map,
        reduce_fn=_hessian_red,
        parallel_iterations=1,
    )
    return H[0]


def hessian_nb_glm(
        batched_data: tf.data.Dataset,
        singleobs_data: tf.data.Dataset,
        sample_indices: tf.Tensor,
        constraints_loc,
        constraints_scale,
        model_vars,
        dtype,
        mode="obs"
):
    """
    Compute the nb_glm model hessian.

    :param data: Dataset iterator over mini-batches of data (used for training).
    :param data: Dataset iterator over single observation batches of data.
    :param sample_indices: Indices of samples to be used.
    :param constraints_loc: Constraints for location model.
        Array with constraints in rows and model parameters in columns.
        Each constraint contains non-zero entries for the a of parameters that 
        has to sum to zero. This constraint is enforced by binding one parameter
        to the negative sum of the other parameters, effectively representing that
        parameter as a function of the other parameters. This dependent
        parameter is indicated by a -1 in this array, the independent parameters
        of that constraint (which may be dependent at an earlier constraint)
        are indicated by a 1.
    :param constraints_scale: Constraints for scale model.
        Array with constraints in rows and model parameters in columns.
        Each constraint contains non-zero entries for the a of parameters that 
        has to sum to zero. This constraint is enforced by binding one parameter
        to the negative sum of the other parameters, effectively representing that
        parameter as a function of the other parameters. This dependent
        parameter is indicated by a -1 in this array, the independent parameters
        of that constraint (which may be dependent at an earlier constraint)
        are indicated by a 1.
    :param mode: str
        Mode by with which hessian is to be evaluated, 
        for analytic solutions of the hessian one can either chose by
        "feature" or by "obs" (observation). Note that sparse
        observation matrices X are often csr, ie. slicing is 
        faster by row/observation, so that hessian evaluation
        by observation is much faster. "tf" allows for
        evaluation of the hessian via the tf.hessian function,
        which is done by feature for implementation reasons.
    """
    if constraints_loc != None and mode != "tf":
        raise ValueError("closed form hessian does not work if constraints_loc is not None")
    if constraints_scale != None and mode != "tf":
        raise ValueError("closed form hessian does not work if constraints_scale is not None")

    if mode == "obs":
        H = _hessian_nb_glm_byobs(
            batched_data=singleobs_data,
            sample_indices=sample_indices,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            model_vars=model_vars,
            dtype=dtype
        )
    elif mode == "feature":
        H = _hessian_nb_glm_byfeature(
            batched_data=batched_data,
            sample_indices=sample_indices,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            model_vars=model_vars,
            dtype=dtype
        )
    elif mode == "tf":
        H = _tf_hessian_nb_glm_byfeature(
            batched_data=batched_data,
            sample_indices=sample_indices,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            model_vars=model_vars,
            dtype=dtype
        )
    else:
        raise ValueError("mode not recognized in hessian_nb_glm: " + mode)

    return H


def param_bounds(dtype):
    if isinstance(dtype, tf.DType):
        min = dtype.min
        max = dtype.max
        dtype = dtype.as_numpy_dtype
    else:
        dtype = np.dtype(dtype)
        min = np.finfo(dtype).min
        max = np.finfo(dtype).max

    sf = dtype(pkg_constants.ACCURACY_MARGIN_RELATIVE_TO_LIMIT)
    bounds_min = {
        "a": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
        "b": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
        "log_mu": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
        "log_r": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
        "mu": np.nextafter(0, np.inf, dtype=dtype),
        "r": np.nextafter(0, np.inf, dtype=dtype),
        "probs": dtype(0),
        "log_probs": np.log(np.nextafter(0, np.inf, dtype=dtype)),
    }
    bounds_max = {
        "a": np.nextafter(np.log(max), -np.inf, dtype=dtype) / sf,
        "b": np.nextafter(np.log(max), -np.inf, dtype=dtype) / sf,
        "log_mu": np.nextafter(np.log(max), -np.inf, dtype=dtype) / sf,
        "log_r": np.nextafter(np.log(max), -np.inf, dtype=dtype) / sf,
        "mu": np.nextafter(max, -np.inf, dtype=dtype) / sf,
        "r": np.nextafter(max, -np.inf, dtype=dtype) / sf,
        "probs": dtype(1),
        "log_probs": dtype(0),
    }
    return bounds_min, bounds_max


def clip_param(param, name):
    bounds_min, bounds_max = param_bounds(param.dtype)
    return tf.clip_by_value(
        param,
        bounds_min[name],
        bounds_max[name]
    )


def apply_constraints(constraints: np.ndarray, var: tf.Variable, dtype: str):
    """ Iteratively build depend variables from other variables via constraints

    :param constraints: Array with constraints in rows and model parameters in columns.
        Each constraint contains non-zero entries for the a of parameters that 
        has to sum to zero. This constraint is enforced by binding one parameter
        to the negative sum of the other parameters, effectively representing that
        parameter as a function of the other parameters. This dependent
        parameter is indicated by a -1 in this array, the independent parameters
        of that constraint (which may be dependent at an earlier constraint)
        are indicated by a 1.
    :param var: Variable tensor features x independent parameters.
    :param dtype: Precision used in tensorflow.

    :return: Full model parameter matrix with dependent parameters.
    """

    # Find all independent variables:
    idx_indep = np.where(np.any(constraints == -1, axis=0) == False)[0]
    # Relate constraints to dependent variables:
    idx_dep = np.array([np.where(constr == -1)[0] for constr in constraints])
    # Only choose dependent variable which was not already defined above:
    idx_dep = np.concatenate([
        x[[xx not in np.concatenate(idx_dep[:i]) for xx in x]] if i > 0 else x
        for i, x in enumerate(idx_dep)
    ])

    # Add column with dependent parameters successfully to
    # the right side of the parameter tensor x. The parameter
    # tensor is initialised with the independent variables var
    # and is grown by one varibale in each iteration until
    # all variables are there.
    x = var
    for i in range(constraints.shape[0]):
        idx_var_i = np.concatenate([idx_indep, idx_dep[:i]])
        constraint_model = constraints[[i], :][:, idx_var_i]
        constraint_model = tf.convert_to_tensor(-constraint_model, dtype=dtype)
        # Compute new dependent variable based on current constrained
        # and add to parameter tensor:
        x = tf.concat([x, tf.matmul(constraint_model, x)], axis=0)

    # Rearrange parameter matrix to follow parameter ordering
    # in design matrix.

    # Assemble index reordering vector:
    idx_var = np.argsort(np.concatenate([idx_indep, idx_dep]))
    # Reorder parameter tensor:
    x = tf.gather(x, indices=idx_var, axis=0)

    return x


class BasicModelGraph:

    def __init__(
            self,
            X,
            design_loc,
            design_scale,
            constraints_loc,
            constraints_scale,
            a,
            b,
            dtype,
            size_factors=None
    ):
        dist_estim = nb_utils.NegativeBinomial(mean=tf.exp(tf.gather(a, 0)),
                                               r=tf.exp(tf.gather(b, 0)),
                                               name="dist_estim")

        # Define first layer of computation graph on identifiable variables
        # to yield dependent set of parameters of model for each location
        # and scale model.

        if constraints_loc is not None:
            a = apply_constraints(constraints_loc, a, dtype=dtype)

        if constraints_scale is not None:
            b = apply_constraints(constraints_scale, b, dtype=dtype)

        with tf.name_scope("mu"):
            log_mu = tf.matmul(design_loc, a, name="log_mu_obs")
            if size_factors is not None:
                log_mu = tf.add(log_mu, size_factors)
            log_mu = clip_param(log_mu, "log_mu")
            mu = tf.exp(log_mu)

        with tf.name_scope("r"):
            log_r = tf.matmul(design_scale, b, name="log_r_obs")
            log_r = clip_param(log_r, "log_r")
            r = tf.exp(log_r)

        dist_obs = nb_utils.NegativeBinomial(mean=mu, r=r, name="dist_obs")

        with tf.name_scope("probs"):
            probs = dist_obs.prob(X)
            probs = clip_param(probs, "probs")

        with tf.name_scope("log_probs"):
            log_probs = dist_obs.log_prob(X)
            log_probs = clip_param(log_probs, "log_probs")

        self.X = X
        self.design_loc = design_loc
        self.design_scale = design_scale

        self.dist_estim = dist_estim
        self.mu_estim = dist_estim.mean()
        self.r_estim = dist_estim.r
        self.sigma2_estim = dist_estim.variance()

        self.dist_obs = dist_obs
        self.mu = mu
        self.r = r
        self.sigma2 = dist_obs.variance()

        self.probs = probs
        self.log_probs = log_probs
        self.log_likelihood = tf.reduce_sum(self.log_probs, axis=0, name="log_likelihood")
        self.norm_log_likelihood = tf.reduce_mean(self.log_probs, axis=0, name="log_likelihood")
        self.norm_neg_log_likelihood = - self.norm_log_likelihood

        with tf.name_scope("loss"):
            self.loss = tf.reduce_sum(self.norm_neg_log_likelihood)


class ModelVars:
    a: tf.Tensor
    b: tf.Tensor
    a_var: tf.Variable
    b_var: tf.Variable
    params: tf.Variable
    """ Build tf.Variables to be optimzed and their constraints.

    a_var and b_var slices of the tf.Variable params which contains
    all parameters to be optimzed during model estimation. 
    Params is defined across both location and scale model so that 
    the hessian can be computed for the entire model.
    a and b are the clipped parameter values which also contain
    constraints and constrained dependent coefficients which are not
    directrly optimzed.
    """

    def __init__(
            self,
            init_dist: nb_utils.NegativeBinomial,
            dtype,
            num_design_loc_params,
            num_design_scale_params,
            num_features,
            init_a=None,
            init_b=None,
            constraints_loc=None,
            constraints_scale=None,
            name="ModelVars",
    ):
        with tf.name_scope(name):
            with tf.name_scope("initialization"):

                if init_a is None:
                    # initialize with observed mean over all observations
                    intercept = tf.log(init_dist.mean())
                    slope = tf.random_uniform(
                        tf.TensorShape([num_design_loc_params - 1, num_features]),
                        minval=np.nextafter(0, 1, dtype=dtype.as_numpy_dtype),
                        maxval=np.sqrt(np.nextafter(0, 1, dtype=dtype.as_numpy_dtype)),
                        dtype=dtype
                    )
                    init_a = tf.concat([
                        intercept,
                        slope,
                    ], axis=-2)
                else:
                    init_a = tf.convert_to_tensor(init_a, dtype=dtype)

                if init_b is None:
                    # initialize with observed variance over all observations
                    intercept = tf.log(init_dist.r)
                    slope = tf.random_uniform(
                        tf.TensorShape([num_design_scale_params - 1, num_features]),
                        minval=np.nextafter(0, 1, dtype=dtype.as_numpy_dtype),
                        maxval=np.sqrt(np.nextafter(0, 1, dtype=dtype.as_numpy_dtype)),
                        dtype=dtype
                    )
                    init_b = tf.concat([
                        intercept,
                        slope,
                    ], axis=-2)
                else:
                    init_b = tf.convert_to_tensor(init_b, dtype=dtype)

                init_a = clip_param(init_a, "a")
                init_b = clip_param(init_b, "b")

            if constraints_loc is not None:
                # Find all dependent variables.
                a_is_dep = np.any(constraints_loc == -1, axis=0)
                # Define reduced variable set which is stucturally identifiable.
                init_a = tf.gather(init_a, indices=np.where(a_is_dep == False)[0], axis=0)

            if constraints_scale is not None:
                # Find all dependent variables.
                b_is_dep = np.any(constraints_scale == -1, axis=0)
                # Define reduced variable set which is stucturally identifiable.
                init_b = tf.gather(init_b, indices=np.where(b_is_dep == False)[0], axis=0)

            # Param is the only tf.Variable in the graph. 
            # a_var and b_var have to be slices of params.
            params = tf.Variable(tf.concat(
                [
                    init_a,
                    init_b,
                ],
                axis=0
            ), name="params")

            a_var = params[0:init_a.shape[0]]
            b_var = params[init_a.shape[0]:]

            # Define first layer of computation graph on identifiable variables
            # to yield dependent set of parameters of model for each location
            # and scale model.

            if constraints_loc is not None:
                a = apply_constraints(constraints_loc, a_var, dtype=dtype)
            else:
                a = a_var

            if constraints_scale is not None:
                b = apply_constraints(constraints_scale, b_var, dtype=dtype)
            else:
                b = b_var

            a_clipped = clip_param(a, "a")
            b_clipped = clip_param(b, "b")

            self.a = a_clipped
            self.b = b_clipped
            self.a_var = a_var
            self.b_var = b_var
            self.params = params


# def feature_wise_bfgs(
#         X,
#         design_loc,
#         design_scale,
#         params,
#         p_shape_a,
#         p_shape_b,
#         size_factors=None
# ) -> List[tf.Tensor]:
#     X_t = tf.transpose(tf.expand_dims(X, axis=0), perm=[2, 0, 1])
#     params_t = tf.transpose(tf.expand_dims(params, axis=0), perm=[2, 0, 1])
#
#     def bfgs(data):  # data is tuple (X_t, a_t, b_t)
#         X_t, a_t, b_t = data
#         X = tf.transpose(X_t)  # observations x features
#         params = tf.transpose(params_t)  # design_params x features
#
#         a_split, b_split = tf.split(params, tf.TensorShape([p_shape_a, p_shape_b]))
#
#         model = BasicModelGraph(X, design_loc, design_scale, a_split, b_split, size_factors=size_factors)
#
#         hess = tf.hessians(model.loss, params)
#
#         def loss_fn(param_vec):
#             a_split, b_split = tf.split(param_vec, tf.TensorShape([p_shape_a, p_shape_b]))
#
#             model = BasicModelGraph(X, design_loc, design_scale, a_split, b_split, size_factors=size_factors)
#
#             return model.loss
#
#         def value_and_grad_fn(param_vec):
#             a_split, b_split = tf.split(param_vec, tf.TensorShape([p_shape_a, p_shape_b]))
#
#             model = BasicModelGraph(X, design_loc, design_scale, a_split, b_split, size_factors=size_factors)
#
#             return model.loss, tf.gradients(model.loss, param_vec)[0]
#
#         bfgs_res = bfgs_minimize(value_and_grad_fn, param_vec, initial_inv_hessian=hess[0])
#
#         return bfgs_res
#
#     bfgs_loop = tf.map_fn(
#         fn=bfgs,
#         elems=(X_t, params_t),
#         dtype=[tf.float32],
#         parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
#     )
#
#     stacked = [tf.squeeze(tf.squeeze(tf.stack(t), axis=2), axis=3) for t in bfgs_loop]
#
#     return stacked


class FullDataModelGraph:
    def __init__(
            self,
            sample_indices: tf.Tensor,
            fetch_fn,
            batch_size: Union[int, tf.Tensor],
            model_vars,
            constraints_loc,
            constraints_scale,
            dtype
    ):
        num_features = model_vars.a.shape[-1]
        dataset = tf.data.Dataset.from_tensor_slices(sample_indices)

        batched_data = dataset.batch(batch_size)
        batched_data = batched_data.map(fetch_fn, num_parallel_calls=pkg_constants.TF_NUM_THREADS)
        batched_data = batched_data.prefetch(1)

        singleobs_data = dataset.map(fetch_fn, num_parallel_calls=pkg_constants.TF_NUM_THREADS)
        singleobs_data = singleobs_data.prefetch(1)


        def map_model(idx, data) -> BasicModelGraph:
            X, design_loc, design_scale, size_factors = data
            model = BasicModelGraph(
                X=X,
                design_loc=design_loc,
                design_scale=design_scale,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                a=model_vars.a_var,
                b=model_vars.b_var,
                dtype=dtype,
                size_factors=size_factors)
            return model

        super()
        model = map_model(*fetch_fn(sample_indices))

        with tf.name_scope("log_likelihood"):
            log_likelihood = op_utils.map_reduce(
                last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
                data=batched_data,
                map_fn=lambda idx, data: map_model(idx, data).log_likelihood,
                parallel_iterations=1,
            )
            norm_log_likelihood = log_likelihood / tf.cast(tf.size(sample_indices), dtype=log_likelihood.dtype)
            norm_neg_log_likelihood = - norm_log_likelihood

        with tf.name_scope("loss"):
            loss = tf.reduce_sum(norm_neg_log_likelihood)

        with tf.name_scope("hessians"):
            hessians = hessian_nb_glm(
                batched_data=batched_data,
                singleobs_data=singleobs_data,
                sample_indices=sample_indices,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                model_vars=model_vars,
                mode=pkg_constants.HESSIAN_MODE,
                dtype=dtype
            )

        self.X = model.X
        self.design_loc = model.design_loc
        self.design_scale = model.design_scale

        self.batched_data = batched_data

        self.dist_estim = model.dist_estim
        self.mu_estim = model.mu_estim
        self.r_estim = model.r_estim
        self.sigma2_estim = model.sigma2_estim

        self.dist_obs = model.dist_obs
        self.mu = model.mu
        self.r = model.r
        self.sigma2 = model.sigma2

        self.probs = model.probs
        self.log_probs = model.log_probs

        # custom
        self.sample_indices = sample_indices

        self.log_likelihood = log_likelihood
        self.norm_log_likelihood = norm_log_likelihood
        self.norm_neg_log_likelihood = norm_neg_log_likelihood
        self.loss = loss

        self.hessians = hessians


class EstimatorGraph(TFEstimatorGraph):
    X: tf.Tensor

    mu: tf.Tensor
    sigma2: tf.Tensor
    a: tf.Tensor
    b: tf.Tensor

    def __init__(
            self,
            fetch_fn,
            feature_isnonzero,
            num_observations,
            num_features,
            num_design_loc_params,
            num_design_scale_params,
            graph: tf.Graph = None,
            batch_size=500,
            init_a=None,
            init_b=None,
            constraints_loc=None,
            constraints_scale=None,
            extended_summary=False,
            dtype="float32"
    ):
        super().__init__(graph)
        self.num_observations = num_observations
        self.num_features = num_features
        self.num_design_loc_params = num_design_loc_params
        self.num_design_scale_params = num_design_scale_params
        self.batch_size = batch_size

        # initial graph elements
        with self.graph.as_default():
            # ### placeholders
            learning_rate = tf.placeholder(dtype, shape=(), name="learning_rate")
            # train_steps = tf.placeholder(tf.int32, shape=(), name="training_steps")

            # ### performance related settings
            buffer_size = 4

            with tf.name_scope("input_pipeline"):
                data_indices = tf.data.Dataset.from_tensor_slices((
                    tf.range(num_observations, name="sample_index")
                ))
                training_data = data_indices.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=2 * batch_size))
                # training_data = training_data.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
                training_data = training_data.batch(batch_size, drop_remainder=True)
                training_data = training_data.map(fetch_fn, num_parallel_calls=pkg_constants.TF_NUM_THREADS)
                training_data = training_data.prefetch(buffer_size)

                iterator = training_data.make_one_shot_iterator()

                batch_sample_index, batch_data = iterator.get_next()
                (batch_X, batch_design_loc, batch_design_scale, batch_size_factors) = batch_data

            dtype = batch_X.dtype

            # implicit broadcasting of X and initial_mixture_probs to
            #   shape (num_mixtures, num_observations, num_features)
            # init_dist = nb_utils.fit(batch_X, axis=-2)
            init_dist = nb_utils.NegativeBinomial(
                mean=tf.random_uniform(
                    minval=10,
                    maxval=1000,
                    shape=[1, num_features],
                    dtype=dtype
                ),
                r=tf.random_uniform(
                    minval=1,
                    maxval=10,
                    shape=[1, num_features],
                    dtype=dtype
                ),
            )
            assert init_dist.r.shape == [1, num_features]

            model_vars = ModelVars(
                init_dist=init_dist,
                dtype=dtype,
                num_design_loc_params=num_design_loc_params,
                num_design_scale_params=num_design_scale_params,
                num_features=num_features,
                init_a=init_a,
                init_b=init_b,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale
            )

            with tf.name_scope("batch"):
                # Batch model:
                #   only `batch_size` observations will be used;
                #   All per-sample variables have to be passed via `data`.
                #   Sample-independent variables (e.g. per-feature distributions) can be created inside the batch model
                batch_model = BasicModelGraph(
                    X=batch_X,
                    design_loc=batch_design_loc,
                    design_scale=batch_design_scale,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    a=model_vars.a_var,
                    b=model_vars.b_var,
                    dtype=dtype,
                    size_factors=batch_size_factors
                )

                # minimize negative log probability (log(1) = 0);
                # use the mean loss to keep a constant learning rate independently of the batch size
                batch_loss = batch_model.loss

            with tf.name_scope("full_data"):
                # ### alternative definitions for custom observations:
                sample_selection = tf.placeholder_with_default(tf.range(num_observations),
                                                               shape=(None,),
                                                               name="sample_selection")
                full_data_model = FullDataModelGraph(
                    sample_indices=sample_selection,
                    fetch_fn=fetch_fn,
                    batch_size=batch_size * buffer_size,
                    model_vars=model_vars,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    dtype=dtype,
                )
                full_data_loss = full_data_model.loss
                fisher_inv = op_utils.pinv(full_data_model.hessians)

                # with tf.name_scope("hessian_diagonal"):
                #     hessian_diagonal = [
                #         tf.map_fn(
                #             # elems=tf.transpose(hess, perm=[2, 0, 1]),
                #             elems=hess,
                #             fn=tf.diag_part,
                #             parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
                #         )
                #         for hess in full_data_model.hessians
                #     ]
                #     fisher_a, fisher_b = hessian_diagonal

                mu = full_data_model.mu
                r = full_data_model.r
                sigma2 = full_data_model.sigma2

            # ### management
            with tf.name_scope("training"):
                global_step = tf.train.get_or_create_global_step()

                a_only_constr = [
                    lambda grad: tf.concat([
                        grad[0:model_vars.a.shape[0]],
                        tf.zeros_like(grad)[model_vars.a.shape[0]:],
                    ], axis=0)
                ]
                b_only_constr = [
                    lambda grad: tf.concat([
                        tf.zeros_like(grad)[0:model_vars.a.shape[0]],
                        grad[model_vars.a.shape[0]:],
                    ], axis=0)
                ]

                # set up trainers for different selections of variables to train
                # set up multiple optimization algorithms for each trainer
                batch_trainers = train_utils.MultiTrainer(
                    loss=batch_model.norm_neg_log_likelihood,
                    variables=[model_vars.params],
                    learning_rate=learning_rate,
                    global_step=global_step,
                    name="batch_trainers"
                )
                batch_trainers_a_only = train_utils.MultiTrainer(
                    loss=batch_model.norm_neg_log_likelihood,
                    # variables=[model_vars.a_var],
                    variables=[model_vars.params],
                    grad_constr=a_only_constr,
                    learning_rate=learning_rate,
                    global_step=global_step,
                    name="batch_trainers_a_only"
                )
                batch_trainers_b_only = train_utils.MultiTrainer(
                    loss=batch_model.norm_neg_log_likelihood,
                    # variables=[model_vars.b_var],
                    variables=[model_vars.params],
                    grad_constr=b_only_constr,
                    learning_rate=learning_rate,
                    global_step=global_step,
                    name="batch_trainers_b_only"
                )

                with tf.name_scope("full_gradient"):
                    batch_gradient = batch_trainers.gradient[0][0]
                    batch_gradient = tf.reduce_sum(tf.abs(batch_gradient), axis=0)

                    # batch_gradient = tf.add_n(
                    #     [tf.reduce_sum(tf.abs(grad), axis=0) for (grad, var) in batch_trainers.gradient])

                full_data_trainers = train_utils.MultiTrainer(
                    loss=full_data_model.norm_neg_log_likelihood,
                    variables=[model_vars.params],
                    learning_rate=learning_rate,
                    global_step=global_step,
                    name="full_data_trainers"
                )
                full_data_trainers_a_only = train_utils.MultiTrainer(
                    loss=full_data_model.norm_neg_log_likelihood,
                    # variables=[model_vars.a_var],
                    variables=[model_vars.params],
                    grad_constr=a_only_constr,
                    learning_rate=learning_rate,
                    global_step=global_step,
                    name="full_data_trainers_a_only"
                )
                full_data_trainers_b_only = train_utils.MultiTrainer(
                    loss=full_data_model.norm_neg_log_likelihood,
                    # variables=[model_vars.b_var],
                    variables=[model_vars.params],
                    grad_constr=b_only_constr,
                    learning_rate=learning_rate,
                    global_step=global_step,
                    name="full_data_trainers_b_only"
                )
                with tf.name_scope("full_gradient"):
                    full_gradient = full_data_trainers.gradient[0][0]
                    full_gradient = tf.reduce_sum(tf.abs(full_gradient), axis=0)
                    # full_gradient = tf.add_n(
                    #     [tf.reduce_sum(tf.abs(grad), axis=0) for (grad, var) in full_data_trainers.gradient])

                with tf.name_scope("newton-raphson"):
                    # tf.gradients(- full_data_model.log_likelihood, [model_vars.a, model_vars.b])
                    param_grad_vec = tf.gradients(- full_data_model.log_likelihood, model_vars.params)[0]
                    param_grad_vec_t = tf.transpose(param_grad_vec)

                    delta_t = tf.squeeze(tf.matrix_solve_ls(
                        # full_data_model.hessians,
                        (full_data_model.hessians + tf.transpose(full_data_model.hessians, perm=[0, 2, 1])) / 2,
                        tf.expand_dims(param_grad_vec_t, axis=-1),
                        fast=False
                    ), axis=-1)
                    delta = tf.transpose(delta_t)
                    nr_update = model_vars.params - learning_rate * delta
                    # nr_update = model_vars.params - delta
                    newton_raphson_op = tf.group(
                        tf.assign(model_vars.params, nr_update),
                        tf.assign_add(global_step, 1)
                    )

                # # ### BFGS implementation using SciPy L-BFGS
                # with tf.name_scope("bfgs"):
                #     feature_idx = tf.placeholder(dtype="int64", shape=())
                #
                #     X_s = tf.gather(X, feature_idx, axis=1)
                #     a_s = tf.gather(a, feature_idx, axis=1)
                #     b_s = tf.gather(b, feature_idx, axis=1)
                #
                #     model = BasicModelGraph(X_s, design_loc, design_scale, a_s, b_s, size_factors=size_factors)
                #
                #     trainer = tf.contrib.opt.ScipyOptimizerInterface(
                #         model.loss,
                #         method='L-BFGS-B',
                #         options={'maxiter': maxiter})

            with tf.name_scope("init_op"):
                init_op = tf.global_variables_initializer()

            # ### output values:
            #       override all-zero features with lower bound coefficients
            with tf.name_scope("output"):
                bounds_min, bounds_max = param_bounds(dtype)

                param_nonzero_a = tf.broadcast_to(feature_isnonzero, [num_design_loc_params, num_features])
                alt_a = tf.concat([
                    # intercept
                    tf.broadcast_to(bounds_min["a"], [1, num_features]),
                    # slope
                    tf.zeros(shape=[num_design_loc_params - 1, num_features], dtype=model_vars.a.dtype),
                ], axis=0, name="alt_a")
                a = tf.where(
                    param_nonzero_a,
                    model_vars.a,
                    alt_a,
                    name="a"
                )

                param_nonzero_b = tf.broadcast_to(feature_isnonzero, [num_design_scale_params, num_features])
                alt_b = tf.concat([
                    # intercept
                    tf.broadcast_to(bounds_max["b"], [1, num_features]),
                    # slope
                    tf.zeros(shape=[num_design_scale_params - 1, num_features], dtype=model_vars.b.dtype),
                ], axis=0, name="alt_b")
                b = tf.where(
                    param_nonzero_b,
                    model_vars.b,
                    alt_b,
                    name="b"
                )

        self.fetch_fn = fetch_fn
        self.model_vars = model_vars
        self.batch_model = batch_model

        self.learning_rate = learning_rate
        self.loss = batch_loss

        self.batch_trainers = batch_trainers
        self.batch_trainers_a_only = batch_trainers_a_only
        self.batch_trainers_b_only = batch_trainers_b_only
        self.full_data_trainers = full_data_trainers
        self.full_data_trainers_a_only = full_data_trainers_a_only
        self.full_data_trainers_b_only = full_data_trainers_b_only
        self.global_step = global_step

        self.gradient = batch_gradient
        # self.gradient_a = batch_gradient_a
        # self.gradient_b = batch_gradient_b

        self.train_op = batch_trainers.train_op_GD

        self.init_ops = []
        self.init_op = init_op

        # # ### set up class attributes
        self.a = a
        self.b = b
        assert (self.a.shape == (num_design_loc_params, num_features))
        assert (self.b.shape == (num_design_scale_params, num_features))

        self.mu = mu
        self.r = r
        self.sigma2 = sigma2

        self.batch_probs = batch_model.probs
        self.batch_log_probs = batch_model.log_probs
        self.batch_log_likelihood = batch_model.norm_log_likelihood

        self.sample_selection = sample_selection
        self.full_data_model = full_data_model

        self.full_loss = full_data_loss

        self.full_gradient = full_gradient
        # self.full_gradient_a = full_gradient_a
        # self.full_gradient_b = full_gradient_b

        # we are minimizing the negative LL instead of maximizing the LL
        # => invert hessians
        self.hessians = - full_data_model.hessians
        self.fisher_inv = fisher_inv

        self.newton_raphson_op = newton_raphson_op

        with tf.name_scope('summaries'):
            tf.summary.histogram('a', model_vars.a)
            tf.summary.histogram('b', model_vars.b)
            tf.summary.scalar('loss', batch_loss)
            tf.summary.scalar('learning_rate', learning_rate)

            if extended_summary:
                tf.summary.scalar('median_ll',
                                  tf.contrib.distributions.percentile(
                                      tf.reduce_sum(batch_model.log_probs, axis=1),
                                      50.)
                                  )
                tf.summary.histogram('gradient_a', tf.gradients(batch_loss, model_vars.a))
                tf.summary.histogram('gradient_b', tf.gradients(batch_loss, model_vars.b))
                tf.summary.histogram("full_gradient", full_gradient)
                tf.summary.scalar("full_gradient_median",
                                  tf.contrib.distributions.percentile(full_gradient, 50.))
                tf.summary.scalar("full_gradient_mean", tf.reduce_mean(full_gradient))

        self.saver = tf.train.Saver()
        self.merged_summary = tf.summary.merge_all()


class Estimator(AbstractEstimator, MonitoredTFEstimator, metaclass=abc.ABCMeta):
    """
    Estimator for Generalized Linear Models (GLMs) with negative binomial noise.
    Uses the natural logarithm as linker function.
    """

    class TrainingStrategy(Enum):
        AUTO = None
        DEFAULT = [
            {
                "learning_rate": 0.1,
                "convergence_criteria": "t_test",
                "stop_at_loss_change": 0.05,
                "loss_window_size": 100,
                "use_batching": True,
                "optim_algo": "ADAM",
            },
            {
                "learning_rate": 0.05,
                "convergence_criteria": "t_test",
                "stop_at_loss_change": 0.05,
                "loss_window_size": 10,
                "use_batching": False,
                "optim_algo": "ADAM",
            },
        ]
        EXACT = [
            {
                "learning_rate": 0.1,
                "convergence_criteria": "t_test",
                "stop_at_loss_change": 0.05,
                "loss_window_size": 100,
                "use_batching": True,
                "optim_algo": "ADAM",
            },
            {
                "learning_rate": 0.05,
                "convergence_criteria": "t_test",
                "stop_at_loss_change": 0.05,
                "loss_window_size": 100,
                "use_batching": True,
                "optim_algo": "ADAM",
            },
            {
                "learning_rate": 0.005,
                "convergence_criteria": "t_test",
                "stop_at_loss_change": 0.25,
                "loss_window_size": 10,
                "use_batching": False,
                "optim_algo": "Newton-Raphson",
            },
        ]
        QUICK = [
            {
                "learning_rate": 0.1,
                "convergence_criteria": "t_test",
                "stop_at_loss_change": 0.05,
                "loss_window_size": 100,
                "use_batching": True,
                "optim_algo": "ADAM",
            },
        ]
        PRE_INITIALIZED = [
            {
                "learning_rate": 0.01,
                "convergence_criteria": "t_test",
                "stop_at_loss_change": 0.25,
                "loss_window_size": 10,
                "use_batching": False,
                "optim_algo": "ADAM",
            },
        ]

    model: EstimatorGraph
    _train_mu: bool
    _train_r: bool

    @classmethod
    def param_shapes(cls) -> dict:
        return ESTIMATOR_PARAMS

    def __init__(
            self,
            input_data: InputData,
            batch_size: int = 500,
            init_model: Model = None,
            graph: tf.Graph = None,
            init_a: Union[np.ndarray, str] = "AUTO",
            init_b: Union[np.ndarray, str] = "AUTO",
            quick_scale: bool = False,
            model: EstimatorGraph = None,
            extended_summary=False,
            dtype="float64",
    ):
        """
        Create a new Estimator

        :param input_data: The input data
        :param batch_size: The batch size to use for minibatch SGD.
            Defaults to '500'
        :param graph: (optional) tf.Graph
        :param init_model: (optional) If provided, this model will be used to initialize this Estimator.
        :param init_a: (Optional) Low-level initial values for a.
            Can be:

            - str:
                * "auto": automatically choose best initialization
                * "random": initialize with random values
                * "init_model": initialize with another model (see `ìnit_model` parameter)
                * "closed_form": try to initialize with closed form
            - np.ndarray: direct initialization of 'a'
        :param init_b: (Optional) Low-level initial values for b
            Can be:

            - str:
                * "auto": automatically choose best initialization
                * "random": initialize with random values
                * "init_model": initialize with another model (see `ìnit_model` parameter)
                * "closed_form": try to initialize with closed form
            - np.ndarray: direct initialization of 'b'
        :param model: (optional) EstimatorGraph to use. Basically for debugging.
        :param quick_scale: `scale` will be fitted faster and maybe less accurate.

        Useful in scenarios where fitting the exact `scale` is not absolutely necessary.
        :param extended_summary: Include detailed information in the summaries.
            Will drastically increase runtime of summary writer, use only for debugging.
        """
        # validate design matrix:
        if np.linalg.matrix_rank(input_data.design_loc) != np.linalg.matrix_rank(input_data.design_loc.T):
            raise ValueError("design_loc matrix is not full rank")
        if np.linalg.matrix_rank(input_data.design_scale) != np.linalg.matrix_rank(input_data.design_scale.T):
            raise ValueError("design_scale matrix is not full rank")

        # ### initialization
        if model is None:
            if graph is None:
                graph = tf.Graph()

            self._input_data = input_data
            self._train_mu = True
            self._train_r = False if quick_scale == True else True

            r"""
            standard:
            Only initialise intercept and keep other coefficients as zero.

            closed-form:
            Initialize with Maximum Likelihood / Maximum of Momentum estimators

            Idea:
            $$
                \theta &= f(x) \\
                \Rightarrow f^{-1}(\theta) &= x \\
                    &= (D \cdot D^{+}) \cdot x \\
                    &= D \cdot (D^{+} \cdot x) \\
                    &= D \cdot x' = f^{-1}(\theta)
            $$
            """
            if input_data.size_factors is not None:
                size_factors_init = np.expand_dims(input_data.size_factors, axis=1)
                size_factors_init = np.broadcast_to(
                    array=size_factors_init,
                    shape=[input_data.size_factors.shape[0], input_data.num_features]
                )
            if isinstance(init_a, str):
                # Chose option if auto was chosen
                if init_a.lower() == "auto":
                    init_a = "closed_form"

                if init_a.lower() == "closed_form":
                    try:
                        unique_design_loc, inverse_idx = np.unique(input_data.design_loc, axis=0, return_inverse=True)
                        if input_data.constraints_loc is not None:
                            unique_design_loc_constraints = input_data.constraints_loc.copy()
                            # -1 in the constraint matrix is used to indicate which variable
                            # is made dependent so that the constrained is fullfilled.
                            # This has to be rewritten here so that the design matrix is full rank
                            # which is necessary so that it can be inverted for parameter
                            # initialisation.
                            unique_design_loc_constraints[unique_design_loc_constraints == -1] = 1
                            # Add constraints into design matrix to remove structural unidentifiability.
                            unique_design_loc = np.vstack([unique_design_loc, unique_design_loc_constraints])
                        if unique_design_loc.shape[1] > matrix_rank(unique_design_loc):
                            logger.warning("Location model is not full rank!")
                        X = input_data.X.assign_coords(group=(("observations",), inverse_idx))
                        if input_data.size_factors is not None:
                            X = np.divide(X, size_factors_init)

                        mean = X.groupby("group").mean(dim="observations")
                        mean = np.nextafter(0, 1, out=mean.values, where=mean == 0, dtype=mean.dtype)

                        a = np.log(mean)
                        if input_data.constraints_loc is not None:
                            a_constraints = np.zeros([input_data.constraints_loc.shape[0], a.shape[1]])
                            # Add constraints (sum to zero) to value vector to remove structural unidentifiability.
                            a = np.vstack([a, a_constraints])
                        # inv_design = np.linalg.pinv(unique_design_loc) # NOTE: this is numerically inaccurate!
                        # inv_design = np.linalg.inv(unique_design_loc) # NOTE: this is exact if full rank!
                        # init_a = np.matmul(inv_design, a)
                        a_prime = np.linalg.lstsq(unique_design_loc, a, rcond=None)
                        init_a = a_prime[0]
                        # stat_utils.rmsd(np.exp(unique_design_loc @ init_a), mean)

                        # train mu, if the closed-form solution is inaccurate
                        self._train_mu = not np.all(a_prime[1] == 0)
                        # Temporal fix: train mu if size factors are given as closed form may be differen:
                        if input_data.size_factors is not None:
                            self._train_mu = True

                        logger.info("Using closed-form MLE initialization for mean")
                        logger.debug("RMSE of closed-form mean:\n%s", a_prime[1])
                        logger.info("Should train mu: %s", self._train_mu)
                    except np.linalg.LinAlgError:
                        logger.warning("Closed form initialization failed!")
                elif init_a.lower() == "standard":
                    mean = np.mean(input_data.X, axis=0)
                    mean = np.nextafter(0, 1, out=mean.values, where=mean == 0, dtype=mean.dtype)
                    init_a = np.zeros([input_data.design_loc.shape[1], input_data.X.shape[1]])
                    init_a[0, :] = np.log(mean)
                    self._train_mu = True

                    logger.info("Using standard initialization for mean")
                    logger.info("Should train mu: %s", self._train_mu)

            if isinstance(init_b, str):
                if init_b.lower() == "auto":
                    init_b = "closed_form"

                if init_b.lower() == "closed_form":
                    try:
                        unique_design_scale, inverse_idx = np.unique(input_data.design_scale, axis=0,
                                                                     return_inverse=True)
                        if input_data.constraints_scale is not None:
                            unique_design_scale_constraints = input_data.constraints_scale.copy()
                            # -1 in the constraint matrix is used to indicate which variable
                            # is made dependent so that the constrained is fullfilled.
                            # This has to be rewritten here so that the design matrix is full rank
                            # which is necessary so that it can be inverted for parameter
                            # initialisation.
                            unique_design_scale_constraints[unique_design_scale_constraints == -1] = 1
                            # Add constraints into design matrix to remove structural unidentifiability.
                            unique_design_scale = np.vstack([unique_design_scale, unique_design_scale_constraints])
                        if unique_design_scale.shape[1] > matrix_rank(unique_design_scale):
                            logger.warning("Scale model is not full rank!")

                        X = input_data.X.assign_coords(group=(("observations",), inverse_idx))
                        if input_data.size_factors is not None:
                            X = np.divide(X, size_factors_init)

                        Xdiff = X - np.exp(input_data.design_loc @ init_a)
                        variance = np.square(Xdiff).groupby("group").mean(dim="observations")

                        group_mean = X.groupby("group").mean(dim="observations")
                        denominator = np.fmax(variance - group_mean, 0)
                        denominator = np.nextafter(0, 1, out=denominator.values, where=denominator == 0,
                                                   dtype=denominator.dtype)
                        r = np.square(group_mean) / denominator
                        r = np.nextafter(0, 1, out=r.values, where=r == 0, dtype=r.dtype)
                        r = np.fmin(r, np.finfo(r.dtype).max)

                        b = np.log(r)
                        if input_data.constraints_scale is not None:
                            b_constraints = np.zeros([input_data.constraints_scale.shape[0], b.shape[1]])
                            # Add constraints (sum to zero) to value vector to remove structural unidentifiability.
                            b = np.vstack([b, b_constraints])
                        # inv_design = np.linalg.pinv(unique_design_scale) # NOTE: this is numerically inaccurate!
                        # inv_design = np.linalg.inv(unique_design_scale) # NOTE: this is exact if full rank!
                        # init_b = np.matmul(inv_design, b)
                        b_prime = np.linalg.lstsq(unique_design_scale, b, rcond=None)
                        init_b = b_prime[0]

                        logger.info("Using closed-form MME initialization for dispersion")
                        logger.debug("RMSE of closed-form dispersion:\n%s", b_prime[1])
                        logger.info("Should train r: %s", self._train_r)
                    except np.linalg.LinAlgError:
                        logger.warning("Closed form initialization failed!")
                elif init_b.lower() == "standard":
                    init_b = np.zeros([input_data.design_scale.shape[1], input_data.X.shape[1]])

                    logger.info("Using standard initialization for dispersion")
                    logger.info("Should train r: %s", self._train_r)

            if init_model is not None:
                if isinstance(init_a, str) and (init_a.lower() == "auto" or init_a.lower() == "init_model"):
                    # location
                    my_loc_names = set(input_data.design_loc_names.values)
                    my_loc_names = my_loc_names.intersection(init_model.input_data.design_loc_names.values)

                    init_loc = np.random.uniform(
                        low=np.nextafter(0, 1, dtype=input_data.X.dtype),
                        high=np.sqrt(np.nextafter(0, 1, dtype=input_data.X.dtype)),
                        size=(input_data.num_design_loc_params, input_data.num_features)
                    )
                    for parm in my_loc_names:
                        init_idx = np.where(init_model.input_data.design_loc_names == parm)
                        my_idx = np.where(input_data.design_loc_names == parm)
                        init_loc[my_idx] = init_model.par_link_loc[init_idx]

                    init_a = init_loc

                if isinstance(init_b, str) and (init_b.lower() == "auto" or init_b.lower() == "init_model"):
                    # scale
                    my_scale_names = set(input_data.design_scale_names.values)
                    my_scale_names = my_scale_names.intersection(init_model.input_data.design_scale_names.values)

                    init_scale = np.random.uniform(
                        low=np.nextafter(0, 1, dtype=input_data.X.dtype),
                        high=np.sqrt(np.nextafter(0, 1, dtype=input_data.X.dtype)),
                        size=(input_data.num_design_scale_params, input_data.num_features)
                    )
                    for parm in my_scale_names:
                        init_idx = np.where(init_model.input_data.design_scale_names == parm)
                        my_idx = np.where(input_data.design_scale_names == parm)
                        init_scale[my_idx] = init_model.par_link_scale[init_idx]

                    init_b = init_scale

        # ### prepare fetch_fn:
        def fetch_fn(idx):
            """
            Documentation of tensorflow coding style in this function:
            tf.py_func defines a python function (the getters of the InputData object slots)
            as a tensorflow operation. Here, the shape of the tensor is lost and
            has to be set with set_shape. For size factors, we use explicit broadcasting
            as explained below.
            """
            # Catch dimension collapse error if idx is only one element long, ie. 0D:
            if len(idx.shape)==0:
                idx = tf.expand_dims(idx, axis=0)

            X_tensor = tf.py_func(
                func=input_data.fetch_X,
                inp=[idx],
                Tout=input_data.X.dtype,
                stateful=False
            )
            X_tensor.set_shape(idx.get_shape().as_list() + [input_data.num_features])
            X_tensor = tf.cast(X_tensor, dtype=dtype)

            design_loc_tensor = tf.py_func(
                func=input_data.fetch_design_loc,
                inp=[idx],
                Tout=input_data.design_loc.dtype,
                stateful=False
            )
            design_loc_tensor.set_shape(idx.get_shape().as_list() + [input_data.num_design_loc_params])
            design_loc_tensor = tf.cast(design_loc_tensor, dtype=dtype)

            design_scale_tensor = tf.py_func(
                func=input_data.fetch_design_scale,
                inp=[idx],
                Tout=input_data.design_scale.dtype,
                stateful=False
            )
            design_scale_tensor.set_shape(idx.get_shape().as_list() + [input_data.num_design_scale_params])
            design_scale_tensor = tf.cast(design_scale_tensor, dtype=dtype)

            if input_data.size_factors is not None:
                size_factors_tensor = tf.log(tf.py_func(
                    func=input_data.fetch_size_factors,
                    inp=[idx],
                    Tout=input_data.size_factors.dtype,
                    stateful=False
                ))
                size_factors_tensor.set_shape(idx.get_shape())
                # Here, we broadcast the size_factor tensor to the batch size,
                # note that this should not consum any more memory than
                # keeping the 1D array and performing implicit broadcasting in 
                # the arithmetic operations in the graph.
                size_factors_tensor = tf.expand_dims(size_factors_tensor, axis=-1)
                size_factors_tensor = tf.cast(size_factors_tensor, dtype=dtype)
            else:
                size_factors_tensor = tf.constant(0, shape=[1, 1], dtype=dtype)
            size_factors_tensor = tf.broadcast_to(size_factors_tensor,
                                                  shape=[tf.size(idx), input_data.num_features])

            # return idx, data
            return idx, (X_tensor, design_loc_tensor, design_scale_tensor, size_factors_tensor)

        if isinstance(init_a, str):
            init_a = None
        else:
            init_a = init_a.astype(dtype)
        if isinstance(init_b, str):
            init_b = None
        else:
            init_b = init_b.astype(dtype)

        with graph.as_default():
            # create model
            model = EstimatorGraph(
                fetch_fn=fetch_fn,
                feature_isnonzero=input_data.feature_isnonzero,
                num_observations=input_data.num_observations,
                num_features=input_data.num_features,
                num_design_loc_params=input_data.num_design_loc_params,
                num_design_scale_params=input_data.num_design_scale_params,
                batch_size=batch_size,
                graph=graph,
                init_a=init_a,
                init_b=init_b,
                constraints_loc=input_data.constraints_loc,
                constraints_scale=input_data.constraints_scale,
                extended_summary=extended_summary,
                dtype=dtype
            )

        MonitoredTFEstimator.__init__(self, model)

    def _scaffold(self):
        with self.model.graph.as_default():
            scaffold = tf.train.Scaffold(
                init_op=self.model.init_op,
                summary_op=self.model.merged_summary,
                saver=self.model.saver,
            )
        return scaffold

    def train(self, *args,
              learning_rate=0.5,
              convergence_criteria="t_test",
              loss_window_size=100,
              stop_at_loss_change=0.05,
              train_mu: bool = None,
              train_r: bool = None,
              use_batching=True,
              optim_algo="gradient_descent",
              **kwargs):
        """
        Starts training of the model

        :param feed_dict: dict of values which will be feeded each `session.run()`

            See also feed_dict parameter of `session.run()`.
        :param learning_rate: learning rate used for optimization
        :param convergence_criteria: criteria after which the training will be interrupted.
            Currently implemented criterias:

            - "simple":
              stop, when `loss(step=i) - loss(step=i-1)` < `stop_at_loss_change`
            - "moving_average":
              stop, when `mean_loss(steps=[i-2N..i-N) - mean_loss(steps=[i-N..i)` < `stop_at_loss_change`
            - "absolute_moving_average":
              stop, when `|mean_loss(steps=[i-2N..i-N) - mean_loss(steps=[i-N..i)|` < `stop_at_loss_change`
            - "t_test" (recommended):
              Perform t-Test between the last [i-2N..i-N] and [i-N..i] losses.
              Stop if P("both distributions are equal") > `stop_at_loss_change`.
        :param stop_at_loss_change: Additional parameter for convergence criteria.

            See parameter `convergence_criteria` for exact meaning
        :param loss_window_size: specifies `N` in `convergence_criteria`.
        :param train_mu: Set to True/False in order to enable/disable training of mu
        :param train_r: Set to True/False in order to enable/disable training of r
        :param use_batching: If True, will use mini-batches with the batch size defined in the constructor.
            Otherwise, the gradient of the full dataset will be used.
        :param optim_algo: name of the requested train op. Can be:

            - "Adam"
            - "Adagrad"
            - "RMSprop"
            - "GradientDescent" or "GD"

            See :func:train_utils.MultiTrainer.train_op_by_name for further details.
        """
        if train_mu is None:
            # check if mu was initialized with MLE
            train_mu = self._train_mu
        if train_r is None:
            # check if r was initialized with MLE
            train_r = self._train_r

        if optim_algo.lower() == "newton-raphson" or \
                optim_algo.lower() == "newton_raphson" or \
                optim_algo.lower() == "nr":
            loss = self.model.full_loss
            train_op = self.model.newton_raphson_op
        elif use_batching:
            loss = self.model.loss
            if train_mu:
                if train_r:
                    train_op = self.model.batch_trainers.train_op_by_name(optim_algo)
                else:
                    train_op = self.model.batch_trainers_a_only.train_op_by_name(optim_algo)
            else:
                if train_r:
                    train_op = self.model.batch_trainers_b_only.train_op_by_name(optim_algo)
                else:
                    logger.info("No training necessary; returning")
                    return
        else:
            loss = self.model.full_loss
            if train_mu:
                if train_r:
                    train_op = self.model.full_data_trainers.train_op_by_name(optim_algo)
                else:
                    train_op = self.model.full_data_trainers_a_only.train_op_by_name(optim_algo)
            else:
                if train_r:
                    train_op = self.model.full_data_trainers_b_only.train_op_by_name(optim_algo)
                else:
                    logger.info("No training necessary; returning")
                    return

        super().train(*args,
                      feed_dict={"learning_rate:0": learning_rate},
                      convergence_criteria=convergence_criteria,
                      loss_window_size=loss_window_size,
                      stop_at_loss_change=stop_at_loss_change,
                      loss=loss,
                      train_op=train_op,
                      **kwargs)

    @property
    def input_data(self):
        return self._input_data

    def train_sequence(self, training_strategy=TrainingStrategy.AUTO):
        if isinstance(training_strategy, Enum):
            training_strategy = training_strategy.value
        elif isinstance(training_strategy, str):
            training_strategy = self.TrainingStrategy[training_strategy].value

        if training_strategy is None:
            if not self._train_mu:
                training_strategy = self.TrainingStrategy.PRE_INITIALIZED.value
            else:
                training_strategy = self.TrainingStrategy.DEFAULT.value

        logger.info("training strategy:\n%s", pprint.pformat(training_strategy))

        for idx, d in enumerate(training_strategy):
            logger.info("Beginning with training sequence #%d", idx + 1)
            self.train(**d)
            logger.info("Training sequence #%d complete", idx + 1)

    # @property
    # def mu(self):
    #     return self.to_xarray("mu")
    #
    # @property
    # def r(self):
    #     return self.to_xarray("r")
    #
    # @property
    # def sigma2(self):
    #     return self.to_xarray("sigma2")

    @property
    def a(self):
        return self.to_xarray("a", coords=self.input_data.data.coords)

    @property
    def b(self):
        return self.to_xarray("b", coords=self.input_data.data.coords)

    @property
    def batch_loss(self):
        return self.to_xarray("loss")

    @property
    def batch_gradient(self):
        return self.to_xarray("gradient", coords=self.input_data.data.coords)

    @property
    def loss(self):
        return self.to_xarray("full_loss")

    @property
    def gradient(self):
        return self.to_xarray("full_gradient", coords=self.input_data.data.coords)

    @property
    def hessians(self):
        return self.to_xarray("hessians", coords=self.input_data.data.coords)

    @property
    def fisher_inv(self):
        return self.to_xarray("fisher_inv", coords=self.input_data.data.coords)

    def finalize(self):
        logger.debug("Collect and compute ouptut")
        store = XArrayEstimatorStore(self)
        logger.debug("Closing session")
        self.close_session()
        return store
