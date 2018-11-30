# from typing import Union, Tuple

import logging
# import pprint
# from enum import Enum

import tensorflow as tf
# import tensorflow_probability as tfp

import numpy as np
import xarray as xr

from .external import AbstractEstimator
from .external import nb_utils, op_utils
from .external import param_bounds as base_param_bounds

logger = logging.getLogger(__name__)

ESTIMATOR_PARAMS = AbstractEstimator.param_shapes().copy()
ESTIMATOR_PARAMS.update({
    "batch_probs": ("mixtures", "batch_observations", "features"),
    "batch_log_probs": ("mixtures", "batch_observations", "features"),
    "batch_log_likelihood": (),
    "full_loss": (),
    "full_gradient": ("features",),
})


def param_bounds(dtype: np.dtype):
    if isinstance(dtype, tf.DType):
        dmin = dtype.min
        dmax = dtype.max
        dtype = dtype.as_numpy_dtype
    else:
        dtype = np.dtype(dtype)
        dmin = np.finfo(dtype).min
        dmax = np.finfo(dtype).max
        dtype = dtype.type

    return base_param_bounds(dtype, dmin, dmax)


def tf_clip_param(param, name):
    bounds_min, bounds_max = param_bounds(param.dtype)
    return tf.clip_by_value(
        param,
        bounds_min[name],
        bounds_max[name]
    )


def np_clip_param(param, name):
    bounds_min, bounds_max = param_bounds(param.dtype)
    if isinstance(param, xr.DataArray):
        return param.clip(
            bounds_min[name],
            bounds_max[name],
            # out=param
        )
    else:
        return np.clip(
            param,
            bounds_min[name],
            bounds_max[name],
            # out=param
        )


class MixtureModel:
    r"""
    Set up the (log-) probability of mixture assignments in a numerically stable way.
    """
    prob: tf.Tensor
    log_prob: tf.Tensor
    logit_prob: tf.Tensor

    def __init__(
            self,
            logits,
            axis=0,
            name="mixture_prob"
    ):
        r"""
        Set up the (log-) probability of mixture assignments in a numerically stable way.

        :param logits: some Tensor or variable containing logits of mixture probabilities
        :param axis: axis along which the sum of mixture probabilities should equal to 100%
        :param name: name scope of the ops
        """
        with tf.name_scope(name):
            # optimize logits to keep `mixture_prob` between the interval [0, 1]
            # use softmax to make sure the mixture probabilites for one observation sum up to 1

            prob = tf.nn.softmax(logits, axis=axis, name="prob")
            log_prob = tf.nn.log_softmax(logits, axis=axis, name="log_prob")

            self.prob = prob
            self.log_prob = log_prob
            self.logit_prob = op_utils.logit(prob, name="normalized_logits")
            self.mixture_assignment = tf.argmax(prob, axis=axis)


class BasicModelGraph:

    def __init__(
            self,
            X,  # (observations, features)
            design_loc,  # (observations, design_loc_params)
            design_scale,  # (observations, design_scale_params)
            design_mixture_loc,  # (design_loc_params, mixtures, design_mixture_loc_params)
            design_mixture_scale,  # (design_scale_params, mixtures, design_mixture_scale_params)
            a,  # (design_loc_params, design_mixture_loc_params, features)
            b,  # (design_scale_params, design_mixture_scale_params, features)
            mixture_logits,  # (mixtures, observations)
            mixture_weight_log_constraints=None,  # (mixtures, observations)
            size_factors=None,
    ):
        mixture_model = MixtureModel(logits=mixture_logits, axis=-1)
        log_mixture_weights = mixture_model.log_prob
        mixture_weights = mixture_model.prob

        dist_estim = nb_utils.NegativeBinomial(
            mean=tf.exp(tf.gather(a, 0)),
            r=tf.exp(tf.gather(b, 0)),
            name="dist_estim"
        )

        par_link_loc = tf.einsum('dmp,dpf->mdf', design_mixture_loc, a)
        # => (mixtures, design_loc_params, features)
        par_link_scale = tf.einsum('dmp,dpf->mdf', design_mixture_scale, b)
        # => (mixtures, design_scale_params, features)

        # design_loc_bcast = tf.broadcast_to(
        #     tf.expand_dims(design_loc, axis=0),
        #     shape=tf.broadcast_dynamic_shape(design_loc.shape, par_link_loc.shape)
        # )
        # design_loc_bcast.set_shape([
        #     par_link_loc.shape[0],
        #     design_loc.shape[0],
        #     design_loc.shape[1]
        # ])
        # design_scale_bcast = tf.broadcast_to(
        #     tf.expand_dims(design_scale, axis=0),
        #     shape=tf.broadcast_dynamic_shape(design_scale.shape, par_link_loc.shape)
        # )
        # design_loc_bcast.set_shape([
        #     par_link_loc.shape[0],
        #     design_scale.shape[0],
        #     design_scale.shape[1]
        # ])
        design_loc_bcast = tf.tile(
            tf.expand_dims(design_loc, axis=0),
            multiples=[par_link_loc.shape[0], 1, 1]
        )
        design_scale_bcast = tf.tile(
            tf.expand_dims(design_scale, axis=0),
            multiples=[par_link_scale.shape[0], 1, 1]
        )

        with tf.name_scope("mu"):
            log_mu = tf.matmul(
                design_loc_bcast,
                par_link_loc,
                name="log_mu_obs"
            )
            # log_mu = tf.einsum('mod,dmp,dpf>mof', design_loc, design_mixture_loc, a)
            if size_factors is not None:
                log_mu = log_mu + size_factors
            log_mu = tf_clip_param(log_mu, "log_mu")
            mu = tf.exp(log_mu)

        with tf.name_scope("r"):
            log_r = tf.matmul(
                design_scale_bcast,
                par_link_scale,
                name="log_r_obs"
            )
            # log_r = tf.einsum('mod,dmp,dpf>mof', design_scale, design_mixture_scale, a)
            log_r = tf_clip_param(log_r, "log_r")
            r = tf.exp(log_r)

        dist_obs = nb_utils.NegativeBinomial(mean=mu, r=r, name="dist_obs")

        # calculate probability of observations:
        with tf.name_scope("log_probs"):
            elemwise_log_probs = dist_obs.log_prob(tf.expand_dims(X, 0), name="log_count_probs")
            elemwise_log_probs = tf_clip_param(elemwise_log_probs, "log_probs")

            # add mixture weight constraints if specified
            if mixture_weight_log_constraints is not None:
                elemwise_log_probs = elemwise_log_probs + tf.expand_dims(
                    tf.transpose(mixture_weight_log_constraints),
                    axis=-1
                )
            elemwise_log_probs = elemwise_log_probs + tf.expand_dims(
                tf.transpose(log_mixture_weights),
                axis=-1
            )
        # calculate joint probability of mixture distributions
        with tf.name_scope("joint_log_probs"):
            # sum up: for k in num_mixtures: mixture_prob(k) * P(r_k, mu_k, sample_data)

            joint_log_probs = tf.reduce_logsumexp(
                elemwise_log_probs,
                axis=-3,
                # name="joint_log_probs"
            )
            joint_log_probs = tf_clip_param(joint_log_probs, "log_probs")

        # with tf.name_scope("probs"):
        #     probs = dist_obs.prob(X)
        #     probs = tf_clip_param(probs, "probs")
        #
        # with tf.name_scope("log_probs"):
        #     log_probs = dist_obs.log_prob(X)
        #     log_probs = tf_clip_param(log_probs, "log_probs")

        with tf.name_scope("estimated_mixture_log_prob"):
            expected_mixture_log_prob = tf.reduce_sum(elemwise_log_probs, axis=-1)
            expected_mixture_log_prob = tf.nn.log_softmax(expected_mixture_log_prob, axis=0)
            expected_mixture_log_prob = tf_clip_param(expected_mixture_log_prob, "mixture_log_prob")

        expected_mixture_prob = tf.exp(expected_mixture_log_prob, name="estimated_mixture_prob")

        # update_mixture_weights_op = tf.assign(mixture_logits, estimated_mixture_log_prob)

        self.X = X
        self.a = a
        self.b = b
        self.par_link_loc = par_link_loc
        self.par_link_scale = par_link_scale
        self.design_loc = design_loc
        self.design_scale = design_scale
        self.design_mixture_loc = design_mixture_loc
        self.design_mixture_scale = design_mixture_scale

        self.mixture_model = mixture_model
        self.log_mixture_weights = log_mixture_weights
        self.mixture_weights = mixture_weights
        self.expected_mixture_log_prob = expected_mixture_log_prob
        self.expected_mixture_prob = expected_mixture_prob
        # self.update_mixture_weights_op = update_mixture_weights_op

        self.dist_estim = dist_estim
        self.mu_estim = dist_estim.mean()
        self.r_estim = dist_estim.r
        self.sigma2_estim = dist_estim.variance()

        self.dist_obs = dist_obs
        self.mu = mu
        self.r = r
        self.sigma2 = dist_obs.variance()

        self.elemwise_probs = tf.exp(elemwise_log_probs, name="probs")
        self.elemwise_log_probs = elemwise_log_probs
        self.log_probs = joint_log_probs
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
    mixture_logits: tf.Variable

    # params: tf.Variable

    def __init__(
            self,
            init_dist: nb_utils.NegativeBinomial,
            dtype,
            num_mixtures,
            num_observations,
            num_features,
            num_design_loc_params,
            num_design_scale_params,
            num_design_mixture_loc_params,
            num_design_mixture_scale_params,
            design_mixture_loc,
            design_mixture_scale,
            init_a=None,
            init_b=None,
            init_mixture_weights=None,
            mixture_weight_constraints=None,
            name="ModelVars",
    ):
        with tf.name_scope(name):
            with tf.name_scope("initialization"):
                # implicit broadcasting of X and initial_mixture_probs to
                #   shape (num_mixtures, num_observations, num_features)
                # init_dist = nb_utils.fit(X, axis=-2)
                # assert init_dist.r.shape == [1, num_features]

                if init_a is None:
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

                    # broadcast along mixture design params
                    init_a = tf.broadcast_to(
                        tf.expand_dims(init_a, -2),
                        shape=[num_design_loc_params, num_design_mixture_loc_params, num_features]
                    )
                else:
                    init_a = tf.convert_to_tensor(init_a, dtype=dtype)

                if init_b is None:
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

                    # broadcast along mixture design params
                    init_b = tf.broadcast_to(
                        tf.expand_dims(init_b, -2),
                        shape=[num_design_scale_params, num_design_mixture_scale_params, num_features]
                    )
                else:
                    init_b = tf.convert_to_tensor(init_b, dtype=dtype)

                init_a = tf_clip_param(init_a, "a")
                init_b = tf_clip_param(init_b, "b")

                if init_mixture_weights is None:
                    init_mixture_weights = tf.random_uniform((num_observations, num_mixtures), 0, 1, dtype=dtype)
                    # make sure the probabilities sum up to 1
                    init_mixture_weights = tf.div(
                        init_mixture_weights,
                        tf.reduce_sum(init_mixture_weights, axis=0, keepdims=True)
                    )
                else:
                    init_mixture_weights = tf.convert_to_tensor(init_mixture_weights, dtype=dtype)

                init_mixture_logits = tf.log(init_mixture_weights, name="init_mixture_logits")
                init_mixture_logits = tf.where(
                    condition=tf.is_nan(init_mixture_logits),
                    x=tf.broadcast_to(np.log(0.5), init_mixture_logits.shape),
                    y=init_mixture_logits
                )
                init_mixture_logits = tf_clip_param(init_mixture_logits, "mixture_logits")

            # params = tf.Variable(tf.concat(
            #     [
            #         init_a,
            #         init_b,
            #     ],
            #     axis=0
            # ), name="params")
            # a_var = params[0:init_a.shape[0]]
            # b_var = params[init_a.shape[0]:]
            #
            # assert a_var.shape == (num_design_loc_params, num_features)
            # assert b_var.shape == (num_design_scale_params, num_features)
            self.design_mixture_loc = tf.identity(design_mixture_loc, name="design_mixture_loc")
            self.design_mixture_scale = tf.identity(design_mixture_scale, name="design_mixture_scale")

            a_var = tf.Variable(init_a, name="a")
            b_var = tf.Variable(init_b, name="b")
            mixture_logits = tf.Variable(init_mixture_logits, name="mixture_logits")

            a_clipped = tf_clip_param(a_var, "a")
            b_clipped = tf_clip_param(b_var, "b")
            mixture_logits_clipped = tf_clip_param(mixture_logits, "mixture_logits")

            self.a = a_clipped
            self.b = b_clipped
            self.mixture_logits = mixture_logits_clipped
            self.a_var = a_var
            self.b_var = b_var
            self.mixture_logits_var = mixture_logits

            if mixture_weight_constraints is not None:
                mixture_weight_constraints = np.asarray(mixture_weight_constraints, dtype=dtype.as_numpy_dtype)
                with tf.name_scope("mixture_weight_constraints"):
                    mixture_weight_constraints = tf.identity(
                        mixture_weight_constraints,
                        name="mixture_weight_constraints"
                    )
                    mixture_weight_constraints = tf_clip_param(mixture_weight_constraints, "mixture_weight_constraints")

                mixture_weight_log_constraints = tf.log(mixture_weight_constraints,
                                                        name="mixture_weight_log_constraints")
            else:
                mixture_weight_log_constraints = None

            self.mixture_weight_constraints = mixture_weight_constraints
            self.mixture_weight_log_constraints = mixture_weight_log_constraints
