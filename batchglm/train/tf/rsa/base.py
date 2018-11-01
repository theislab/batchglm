# from typing import Union, Tuple

import logging
# import pprint
# from enum import Enum

import tensorflow as tf
# import tensorflow_probability as tfp

import numpy as np

from .external import AbstractEstimator
from .external import nb_utils, op_utils
from .external import pkg_constants

logger = logging.getLogger(__name__)

ESTIMATOR_PARAMS = AbstractEstimator.param_shapes().copy()
ESTIMATOR_PARAMS.update({
    "batch_probs": ("mixtures", "batch_observations", "features"),
    "batch_log_probs": ("mixtures", "batch_observations", "features"),
    "batch_log_likelihood": (),
    "full_loss": (),
    "full_gradient": ("features",),
})


def param_bounds(dtype):
    if isinstance(dtype, tf.DType):
        dmin = dtype.min
        dmax = dtype.max
        dtype = dtype.as_numpy_dtype
    else:
        dtype = np.dtype(dtype)
        dmin = np.finfo(dtype).min
        dmax = np.finfo(dtype).max
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
        "mixture_prob": dtype(0),
        "mixture_log_prob": np.log(np.nextafter(0, np.inf, dtype=dtype)),
        "mixture_logits": np.log(np.nextafter(0, np.inf, dtype=dtype)),
    }
    bounds_max = {
        "a": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
        "b": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
        "log_mu": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
        "log_r": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
        "mu": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
        "r": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
        "probs": dtype(1),
        "log_probs": dtype(0),
        "mixture_prob": dtype(1),
        "mixture_log_prob": dtype(0),
        "mixture_logits": dtype(0),
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


class MixtureModel:
    r"""
    Set up the (log-) probability of mixture assignments in a numerically stable way.
    """
    prob: tf.Tensor
    log_prob: tf.Tensor
    logit_prob: tf.Tensor

    def __init__(self, logits, axis=0, name="mixture_prob"):
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
            self.normalized_logits = op_utils.logit(prob, name="normalized_logits")
            self.mixture_assignment = tf.argmax(prob, axis=0)


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
            size_factors=None,
    ):
        mixture_model = MixtureModel(logits=mixture_logits)
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

        with tf.name_scope("mu"):
            log_mu = tf.matmul(tf.expand_dims(design_loc, axis=0), par_link_loc, name="log_mu_obs")
            # log_mu = tf.einsum('mod,dmp,dpf>mof', design_loc, design_mixture_loc, a)
            if size_factors is not None:
                log_mu = log_mu + size_factors
            log_mu = tf_clip_param(log_mu, "log_mu")
            mu = tf.exp(log_mu)

        with tf.name_scope("r"):
            log_r = tf.matmul(tf.expand_dims(design_scale, axis=0), par_link_scale, name="log_r_obs")
            # log_r = tf.einsum('mod,dmp,dpf>mof', design_scale, design_mixture_scale, a)
            log_r = tf_clip_param(log_r, "log_r")
            r = tf.exp(log_r)

        dist_obs = nb_utils.NegativeBinomial(mean=mu, r=r, name="dist_obs")

        # calculate probability of observations:
        log_probs = dist_obs.log_prob(tf.expand_dims(X, 0), name="log_count_probs")

        # calculate joint probability of mixture distributions
        with tf.name_scope("joint_log_probs"):
            # sum up: for k in num_mixtures: mixture_prob(k) * P(r_k, mu_k, sample_data)
            joint_log_probs = tf.reduce_logsumexp(
                log_probs + tf.expand_dims(log_mixture_weights, -1),
                axis=-3,
                # name="joint_log_probs"
            )
            joint_log_probs = tf_clip_param(joint_log_probs, "joint_log_probs")

        # with tf.name_scope("probs"):
        #     probs = dist_obs.prob(X)
        #     probs = tf_clip_param(probs, "probs")
        #
        # with tf.name_scope("log_probs"):
        #     log_probs = dist_obs.log_prob(X)
        #     log_probs = tf_clip_param(log_probs, "log_probs")

        self.X = X
        self.a = a
        self.b = b
        self.design_loc = design_loc
        self.design_scale = design_scale
        self.design_mixture_loc = design_mixture_loc
        self.design_mixture_scale = design_mixture_scale

        self.mixture_model = mixture_model
        self.log_mixture_weights = log_mixture_weights
        self.mixture_weights = mixture_weights

        self.dist_estim = dist_estim
        self.mu_estim = dist_estim.mean()
        self.r_estim = dist_estim.r
        self.sigma2_estim = dist_estim.variance()

        self.dist_obs = dist_obs
        self.mu = mu
        self.r = r
        self.sigma2 = dist_obs.variance()

        self.probs = tf.exp(log_probs, name="probs")
        self.log_probs = log_probs
        self.joint_log_probs = joint_log_probs
        self.log_likelihood = tf.reduce_sum(self.joint_log_probs, axis=0, name="log_likelihood")
        self.norm_log_likelihood = tf.reduce_mean(self.joint_log_probs, axis=0, name="log_likelihood")
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
            init_mixture_probs=None,
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
                else:
                    init_b = tf.convert_to_tensor(init_b, dtype=dtype)

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

                if init_mixture_probs is None:
                    init_mixture_probs = tf.random_uniform((num_mixtures, num_observations), 0, 1, dtype=tf.float32)
                    # make sure the probabilities sum up to 1
                    init_mixture_probs = tf.div(
                        init_mixture_probs,
                        tf.reduce_sum(init_mixture_probs, axis=0, keepdims=True)
                    )

                init_mixture_logits = tf.log(init_mixture_probs, name="init_mixture_logits")
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
            # self.params = params

# class LinearBatchModel:
#     a: tf.Tensor
#     a_intercept: tf.Variable
#     a_slope: tf.Variable
#     b: tf.Tensor
#     b_intercept: tf.Variable
#     b_slope: tf.Variable
#     log_mu_obs: tf.Tensor
#     log_r_obs: tf.Tensor
#     log_count_probs: tf.Tensor
#     joint_log_probs: tf.Tensor
#     loss: tf.Tensor
#
#     def __init__(self,
#                  init_dist: nb_utils.NegativeBinomial,
#                  sample_index,
#                  sample_data,
#                  design,
#                  mixture_model: MixtureModel,
#                  name="Linear_Batch_Model"):
#         with tf.name_scope(name):
#             num_mixtures = mixture_model.prob.shape[0]
#             num_design_params = design.shape[-1]
#             (batch_size, num_features) = sample_data.shape
#
#             mixture_log_prob = tf.gather(mixture_model.log_prob, sample_index, axis=-1)
#
#             assert sample_data.shape == [batch_size, num_features]
#             assert design.shape == [batch_size, num_design_params]
#             assert mixture_log_prob.shape == [num_mixtures, batch_size]
#
#             with tf.name_scope("initialization"):
#                 init_a_intercept = tf.log(init_dist.mean())
#                 init_b_intercept = tf.log(init_dist.r)
#
#                 assert init_a_intercept.shape == [num_mixtures, 1, num_features] == init_b_intercept.shape
#
#                 init_a_slopes = tf.log(tf.random_uniform(
#                     tf.TensorShape([1, num_design_params - 1, num_features]),
#                     maxval=0.1,
#                     dtype=design.dtype
#                 ))
#
#                 init_b_slopes = init_a_slopes
#
#             a, a_intercept, a_slope = tf_linreg.param_variable(init_a_intercept, init_a_slopes, name="a")
#             b, b_intercept, b_slope = tf_linreg.param_variable(init_b_intercept, init_b_slopes, name="b")
#             assert a.shape == (num_mixtures, num_design_params, num_features) == b.shape
#
#             dist_estim = nb_utils.NegativeBinomial(mean=tf.exp(a_intercept),
#                                                    r=tf.exp(b_intercept),
#                                                    name="dist_estim")
#
#             with tf.name_scope("broadcast"):
#                 design = tf.expand_dims(design, axis=0)
#             design = tf.tile(design, (num_mixtures, 1, 1))
#             assert (design.shape == (num_mixtures, batch_size, num_design_params))
#
#             with tf.name_scope("mu"):
#                 log_mu = tf.matmul(design, a, name="log_mu_obs")
#                 log_mu = tf.clip_by_value(log_mu, log_mu.dtype.min, log_mu.dtype.max)
#                 mu = tf.exp(log_mu)
#
#             with tf.name_scope("r"):
#                 log_r = tf.matmul(design, b, name="log_r_obs")
#                 log_r = tf.clip_by_value(log_r, log_r.dtype.min, log_r.dtype.max)
#                 r = tf.exp(log_r)
#
#             dist_obs = nb_utils.NegativeBinomial(r=r, mean=mu, name="dist_obs")
#
#             # calculate mixture model probability:
#             log_count_probs = dist_obs.log_prob(tf.expand_dims(sample_data, 0), name="log_count_probs")
#
#             # sum up: for k in num_mixtures: mixture_prob(k) * P(r_k, mu_k, sample_data)
#             joint_log_probs = tf.reduce_logsumexp(log_count_probs + tf.expand_dims(mixture_log_prob, -1),
#                                                   axis=-3,
#                                                   name="joint_log_probs")
#
#             # probs = tf.exp(joint_log_probs, name="joint_probs")
#
#             # minimize negative log probability (log(1) = 0);
#             # use the mean loss to keep a constant learning rate independently of the batch size
#             loss = -tf.reduce_mean(joint_log_probs, name="loss")
#
#             self.a = a
#             self.a_intercept = a_intercept
#             self.a_slope = a_slope
#             self.b = b
#             self.b_intercept = b_intercept
#             self.b_slope = b_slope
#             self.dist_estim = dist_estim
#             self.dist_obs = dist_obs
#             self.log_mu_obs = log_mu
#             self.log_r_obs = log_r
#             self.log_count_probs = log_count_probs
#             self.joint_log_probs = joint_log_probs
#             self.loss = loss
