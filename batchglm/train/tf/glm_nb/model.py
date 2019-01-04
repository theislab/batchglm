import logging

import tensorflow as tf

import numpy as np

from .external import ProcessModelGLM, ModelVarsGLM, BasicModelGraphGLM
from .external import pkg_constants

logger = logging.getLogger(__name__)


class ProcessModel(ProcessModelGLM):

    def param_bounds(
            self,
            dtype
    ):
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
        }
        return bounds_min, bounds_max


class ModelVars(ProcessModel, ModelVarsGLM):
    """
    Full class.
    """


class BasicModelGraph(ProcessModel, BasicModelGraphGLM):

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
        """

        :param X: tensor (observations x features)
            The input data.
        :param design_loc: Some matrix format (observations x mean model parameters)
            The location design model. Optional if already specified in `data`
        :param design_scale: Some matrix format (observations x dispersion model parameters)
            The scale design model. Optional if already specified in `data`
        :param constraints_loc: tensor (all parameters x dependent parameters)
            Tensor that encodes how complete parameter set which includes dependent
            parameters arises from indepedent parameters: all = <constraints, indep>.
            This tensor describes this relation for the mean model.
            This form of constraints is used in vector generalized linear models (VGLMs).
        :param constraints_scale: tensor (all parameters x dependent parameters)
            Tensor that encodes how complete parameter set which includes dependent
            parameters arises from indepedent parameters: all = <constraints, indep>.
            This tensor describes this relation for the dispersion model.
            This form of constraints is used in vector generalized linear models (VGLMs).
        :param b: tf.Variable or tensor (dispersion model size x features)
            Dispersion model variables.
        :param dtype: Precision used in tensorflow.
        :param size_factors: tensor (observations x features)
            Constant scaling factors for mean model, such as library size factors.
        """
        with tf.name_scope("mu"):
            log_mu = tf.matmul(design_loc, tf.matmul(constraints_loc,  a), name="log_mu_obs")
            if size_factors is not None:
                log_mu = tf.add(log_mu, size_factors)
            log_mu = self.tf_clip_param(log_mu, "log_mu")
            mu = tf.exp(log_mu)

        with tf.name_scope("r"):
            log_r = tf.matmul(design_scale, tf.matmul(constraints_scale,  b), name="log_r_obs")
            log_r = self.tf_clip_param(log_r, "log_r")
            r = tf.exp(log_r)

        with tf.name_scope("sigma2"):
            sigma2 = mu + tf.multiply(tf.square(mu), r)

        with tf.name_scope("log_probs"):
            log_r_plus_mu = tf.log(r+mu)
            log_probs = tf.math.lgamma(r+X) - \
                     tf.math.lgamma(X+1) - tf.math.lgamma(r) + \
                     tf.multiply(X, log_mu - log_r_plus_mu) + \
                     tf.multiply(r, log_r - log_r_plus_mu)
            log_probs = self.tf_clip_param(log_probs, "log_probs")

        with tf.name_scope("probs"):
            probs = tf.exp(log_probs)
            probs = self.tf_clip_param(probs, "probs")

        self.X = X
        self.design_loc = design_loc
        self.design_scale = design_scale

        self.mu = mu
        self.r = r
        self.sigma2 = sigma2

        self.probs = probs
        self.log_probs = log_probs
        self.log_likelihood = tf.reduce_sum(self.log_probs, axis=0, name="log_likelihood")
        self.norm_log_likelihood = tf.reduce_mean(self.log_probs, axis=0, name="log_likelihood")
        self.norm_neg_log_likelihood = - self.norm_log_likelihood

        with tf.name_scope("loss"):
            self.loss = tf.reduce_sum(self.norm_neg_log_likelihood)
