from typing import Dict, Any, Union

import xarray as xr
import tensorflow as tf

from .external import op_utils
from .estimator import BasicModelGraph

def _hessian_nb_gl_aa_coef_invariant(
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
    scalar_one = tf.constant(1, shape=[1,1], dtype=dtype)
    const = tf.negative(tf.multiply(mu, # [observations x features]
        tf.divide(
            tf.divide(X,r)+scalar_one,
            tf.square(scalar_one+tf.divide(mu,r))
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
            &*r*\bigg(\psi_0(r+Y)+\psi_0(r)+r*\psi_1(r+Y)+r*\psi_1(r)\\
            &-\frac{mu}{(r+mu)^2}*(r+Y) \\
            &+\frac{mu-r}{r+mu}+\log(r)+1-\log(r+m) \bigg) \\
        const = r*\bigg(\psi_0(r+Y)+\psi_0(r)+r*\psi_1(r+Y)+r*\psi_1(r)\\
            &-\frac{mu}{(r+mu)^2}*(r+Y) \\
            &+\frac{mu-r}{r+mu}+\log(r)+1-\log(r+m) \bigg) \\
        H^{r,r}_{i,j}&= X^r_i*X^r_j * const
    
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
    scalar_one = tf.constant(1, shape=[1,1], dtype=dtype)
    scalar_two = tf.constant(2, shape=[1,1], dtype=dtype)
    const1 = tf.add( # [observations, features]
        tf.math.polygamma(a=scalar_one, x=tf.add(r,X)),
        tf.add(
            tf.math.polygamma(a=scalar_one, x=r),
            tf.add(
                tf.math.polygamma(a=scalar_two, x=tf.add(r,X)),
                tf.math.polygamma(a=scalar_two, x=r)
                )
            )
        )
    const2 = tf.multiply( # [observations, features]
        tf.divide(mu,tf.square(tf.add(r,mu))),
        tf.add(r,X)
        )
    const3 = tf.add( # [observations, features]
        tf.divide(tf.substract(mu,r), tf.add(mu,r)),
        tf.substract(
            tf.add(tf.log(r), scalar_one),
            tf.log(tf.add(r,mu))
            )
        )
    const = tf.multiply(tf.multiply(const1, const2), const3) # [observations, features]
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
        &H^{r,m}_{i,j} = X^m_i*X^r_j*mu*\frac{Y-mu}{(1+mu/r)^2} \\
        &const = mu*\frac{Y-mu}{(1+mu/r)^2} \\
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
    scalar_one = tf.constant(1, shape=[1,1], dtype=dtype)
    const = tf.multiply(mu, # [observations, features]
        tf.divide(
            Y-mu,
            tf.square(scalar_one+tf.divide(mu,r))
            )
        )
    return const


def _hessian_nb_glm_byobs(
    data,
    sample_indices,
    constraints_loc,
    constraints_scale,
    model_vars
):
    """
    Compute the closed-form of the nb_glm model hessian 
    by evalutating its terms grouped by observations.

    Has three subfunctions which built the specific blocks of the hessian
    and one subfunction which concatenates the blocks into a full hessian.
    """

    def _hessian_nb_glm_aa_byobs(X, mu, r):
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
        scalar_one = tf.constant(1, shape=[1,1], dtype=dtype)
        const = _hessian_nb_glm_aa_coef_invariant( # [observations=1 x features]
            X=X,
            mu=mu,
            r=r,
            dtype=dtype
        )
        nonconst = tf.matmul(tf.transpose(design_loc), design_loc) # [coefficients, coefficients]
        nonconst = tf.expand_dims(nonconst, axis=-1) # [coefficients, coefficients, observations=1]
        Hblock = tf.tranpose(tf.tensordot( # [features, coefficients, coefficients]
            a=nonconst, # [coefficients, coefficients, observations=1]
            b=const, # [observations=1 x features]
            axis=1 # collapse last dimension of a and first dimension of b
        ))
        return Hblock


    def _hessian_nb_glm_bb_byobs(X, mu, r):
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
        const = _hessian_nb_glm_bb_coef_invariant( # [observations=1 x features]
            X=X,
            mu=mu,
            r=r,
            dtype=dtype
        )
        nonconst = tf.matmul(tf.transpose(design_scale), design_scale) # [coefficients, coefficients]
        nonconst = tf.expand_dims(nonconst, axis=-1) # [coefficients, coefficients, observations=1]
        Hblock = tf.tranpose(tf.tensordot( # [features, coefficients, coefficients]
            a=nonconst, # [coefficients, coefficients, observations=1]
            b=const, # [observations=1 x features]
            axis=1 # collapse last dimension of a and first dimension of b
        ))
        return Hblock
        
    def _hessian_nb_glm_ab_byobs(X, mu, r):
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
        const = _hessian_nb_glm_ab_coef_invariant( # [observations=1 x features]
            X=X,
            mu=mu,
            r=r,
            dtype=dtype
        )
        nonconst = tf.matmul(tf.transpose(design_loc), design_scale) # [coefficient_loc, coefficients_scale]
        nonconst = tf.expand_dims(nonconst, axis=-1) # [coefficient_loc, coefficients_scale, observations=1]
        Hblock = tf.tranpose(tf.tensordot( # [features, coefficient_loc, coefficients_scale]
            a=nonconst, # [coefficient_loc, coefficients_scale, observations=1]
            b=const, # [observations=1 x features]
            axis=1 # collapse last dimension of a and first dimension of b
        ))
        return Hblock

    def _hessian_nb_glm_assemble_byobs(data):
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
        
        H_aa = _hessian_nb_glm_aa_byobs(X=X, mu=mu, r=r)
        H_bb = _hessian_nb_glm_bb_byobs(X=X, mu=mu, r=r)
        H_ab = _hessian_nb_glm_bb_byobs(X=X, mu=mu, r=r)
        H = tf.concat(
            tf.concat(H_aa, H_ab, axis=1),
            tf.concat(tf.transpose(H_ab), H_bb, axis=1), 
            axis=0
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
        return [tf.add(p, c) for p, c in zip(prev, cur)]

    dtype = data.X.dtype
    params=model_vars.params
    p_shape_a=model_vars.a_var.shape[0]
    p_shape_b=model_vars.b_var.shape[0]

    H = op_utils.map_reduce(
        last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
        data=data,
        map_fn=_hessian_nb_glm_assemble_byobs,
        reduce_fn=hessian_red,
        parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
    )
    return H[0]


def _hessian_nb_glm_byfeature(
    data,
    sample_indices,
    constraints_loc,
    constraints_scale,
    model_vars
):
    """
    Compute the closed-form of the nb_glm model hessian 
    by evalutating its terms grouped by features.


    Has three subfunctions which built the specific blocks of the hessian
    and one subfunction which concatenates the blocks into a full hessian.
    """

    def _hessian_nb_glm_aa_byfeature(X, mu, r):
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
        scalar_one = tf.constant(1, shape=[1,1], dtype=dtype)
        const = _hessian_nb_glm_aa_coef_invariant( # [observations x features=1]
            X=X,
            mu=mu,
            r=r,
            dtype=dtype
        )
        # The second dimension of const is only one element long,
        # this was a feature before but is no recycled into coefficients.
        const = tf.broadcast_to(const, shape=design_scale.T.shape) # [observations, coefficients]
        Hblock = tf.matmul( # [coefficients, coefficients]
            tf.transpose(design_loc), # [coefficients, observations]
            tf.multiply(design_loc, const) # [observations, coefficients]
            axes=1
        )
        # Prepare stacking across first dimension (features):
        Hblock = tf.expand_dims(Hblock, axis=0) # [features=1, coefficients, coefficients]
        return Hblock


    def _hessian_nb_glm_bb_byfeature(X, mu, r):
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
        const = _hessian_nb_glm_bb_coef_invariant( # [observations x features=1]
            X=X,
            mu=mu,
            r=r,
            dtype=dtype
        )
        # The second dimension of const is only one element long,
        # this was a feature before but is no recycled into coefficients.
        const = tf.broadcast_to(const, shape=design_scale.T.shape) # [observations, coefficients]
        Hblock = tf.matmul( # [coefficients, coefficients]
            tf.transpose(design_scale), # [coefficients, observations]
            tf.multiply(design_scale, const) # [observations, coefficients]
            axes=1
        )
        # Prepare stacking across first dimension (features):
        Hblock = tf.expand_dims(Hblock, axis=0) # [features=1, coefficients, coefficients]
        return Hblock

    def _hessian_nb_glm_ab_byfeature(X, mu, r):
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
        const = _hessian_nb_glm_ab_coef_invariant( # [observations x features=1]
            X=X,
            mu=mu,
            r=r,
            dtype=dtype
        )
        # The second dimension of const is only one element long,
        # this was a feature before but is no recycled into coefficients_scale.
        const = tf.broadcast_to(const, shape=design_scale.T.shape) # [observations, coefficients_scale]
        Hblock = tf.matmul( # [coefficients_loc, coefficients_scale]
            tf.transpost(design_loc), # [coefficients_loc, observations]
            tf.multiply(design_scale, const), # [observations, coefficients_scale]
            axes=1
        )
        # Prepare stacking across first dimension (features):
        Hblock = tf.expand_dims(Hblock, axis=0) # [features=1, coefficients_loc, coefficients_scale]
        return Hblock

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
        X = tf.tranpose(X_t)
        size_factors = tf.tranpose(size_factors_t)
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
        
        H_aa = _hessian_nb_glm_aa_byfeature(X=X, mu=mu, r=r)
        H_bb = _hessian_nb_glm_bb_byfeature(X=X, mu=mu, r=r)
        H_ab = _hessian_nb_glm_bb_byfeature(X=X, mu=mu, r=r)
        H = tf.concat(
            tf.concat(H_aa, H_ab, axis=1),
            tf.concat(tf.transpose(H_ab), H_bb, axis=1), 
            axis=0
        )
        return H

    X, design_loc, design_scale, size_factors = data
    dtype = X.dtype
    params=model_vars.params
    p_shape_a=model_vars.a_var.shape[0]
    p_shape_b=model_vars.b_var.shape[0]

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


def _numeric_hessian_nb_glm_byfeature(
    data,
    sample_indices,
    constraints_loc,
    constraints_scale,
    model_vars,
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
            hess = tf.hessians(- model.log_likelihood, params)
            return hess

        # Map hessian computation across genes
        hessians = tf.map_fn(
            fn=hessian,
            elems=(X_t, size_factors_t, params_t),
            dtype=[dtype],  # hessians of [a, b]
            parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
        )
        
        stacked = [tf.squeeze(tf.squeeze(tf.stack(t), axis=2), axis=3) for t in hessians]
        
        return stacked

    def hessian_map(idx, data):
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

    def hessian_red(prev, cur):
        return [tf.add(p, c) for p, c in zip(prev, cur)]

    H = op_utils.map_reduce(
        last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
        data=batched_data,
        map_fn=hessian_map,
        reduce_fn=hessian_red,
        parallel_iterations=1,
    )
    return H[0]


def hessian_nb_glm(
    data: tf.data.Dataset,
    sample_indices: tf.Tensor,
    constraints_loc,
    constraints_scale,
    model_vars,
    mode="obs"
):
    """
    Compute the nb_glm model hessian.

    :param data: Dataset iterator.
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
    if constraints_loc!=None and mode!="tf":
        raise ValueError("closed form hessian does not work if constraints_loc is not None")
    if constraints_scale!=None and mode!="tf":
        raise ValueError("closed form hessian does not work if constraints_scale is not None")

    if mode=="obs"
        H = _hessian_nb_glm_byobs(
            data=data,
            sample_indices=sample_indices,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            model_vars=model_vars
        )
    elif mode=="feature"
        H = _hessian_nb_glm_byfeature(
            data=data,
            sample_indices=sample_indices,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            model_vars=model_vars
        )
    elif mode=="tf"
        H = _numeric_hessian_nb_glm_byfeature(
            data=data,
            sample_indices=sample_indices,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            model_vars=model_vars
        )
    else:
        raise ValueError("mode not recognized in hessian_nb_glm: "+mode)
    return H
