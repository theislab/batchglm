import logging
# import pprint
# from enum import Enum

import tensorflow as tf
# import tensorflow_probability as tfp

import numpy as np

try:
    import anndata
except ImportError:
    anndata = None

from .external import AbstractEstimator
from .external import nb_utils
from .external import pkg_constants

logger = logging.getLogger(__name__)

ESTIMATOR_PARAMS = AbstractEstimator.param_shapes().copy()
ESTIMATOR_PARAMS.update({
    "batch_probs": ("batch_observations", "features"),
    "batch_log_probs": ("batch_observations", "features"),
    "batch_log_likelihood": (),
    "full_loss": (),
    "full_gradient": ("features",),
})


def param_bounds(dtype):
    if isinstance(dtype, tf.DType):
        min = dtype.min
        max = dtype.max
        dtype = dtype.as_numpy_dtype
    else:
        dtype = np.dtype(dtype)
        min = np.finfo(dtype).min
        max = np.finfo(dtype).max
        dtype = dtype.type

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


def tf_clip_param(param, name):
    bounds_min, bounds_max = param_bounds(param.dtype)
    return tf.clip_by_value(
        param,
        bounds_min[name],
        bounds_max[name]
    )


def np_clip_param(param, name):
    bounds_min, bounds_max = param_bounds(param.dtype)
    return np.clip(
        param,
        bounds_min[name],
        bounds_max[name],
        out=param
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
    idx_indep = np.where(np.all(constraints == -1, axis=0))[0]
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
            log_mu = tf_clip_param(log_mu, "log_mu")
            mu = tf.exp(log_mu)

        with tf.name_scope("r"):
            log_r = tf.matmul(design_scale, b, name="log_r_obs")
            log_r = tf_clip_param(log_r, "log_r")
            r = tf.exp(log_r)

        dist_obs = nb_utils.NegativeBinomial(mean=mu, r=r, name="dist_obs")

        with tf.name_scope("probs"):
            probs = dist_obs.prob(X)
            probs = tf_clip_param(probs, "probs")

        with tf.name_scope("log_probs"):
            log_probs = dist_obs.log_prob(X)
            log_probs = tf_clip_param(log_probs, "log_probs")

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

                init_a = tf_clip_param(init_a, "a")
                init_b = tf_clip_param(init_b, "b")

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

            a_clipped = tf_clip_param(a, "a")
            b_clipped = tf_clip_param(b, "b")

            self.a = a_clipped
            self.b = b_clipped
            self.a_var = a_var
            self.b_var = b_var
            self.params = params
