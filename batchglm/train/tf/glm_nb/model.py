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
        :param constraints_loc: np.ndarray (constraints on mean model x mean model parameters)
            Constraints for location model.
            Array with constraints in rows and model parameters in columns.
            Each constraint contains non-zero entries for the a of parameters that
            has to sum to zero. This constraint is enforced by binding one parameter
            to the negative sum of the other parameters, effectively representing that
            parameter as a function of the other parameters. This dependent
            parameter is indicated by a -1 in this array, the independent parameters
            of that constraint (which may be dependent at an earlier constraint)
            are indicated by a 1.
        :param constraints_scale: np.ndarray (constraints on mean model x mean model parameters)
            Constraints for scale model.
            Array with constraints in rows and model parameters in columns.
            Each constraint contains non-zero entries for the a of parameters that
            has to sum to zero. This constraint is enforced by binding one parameter
            to the negative sum of the other parameters, effectively representing that
            parameter as a function of the other parameters. This dependent
            parameter is indicated by a -1 in this array, the independent parameters
            of that constraint (which may be dependent at an earlier constraint)
            are indicated by a 1.
        :param b: tf.Variable or tensor (dispersion model size x features)
            Dispersion model variables.
        :param dtype: Precision used in tensorflow.
        :param size_factors: tensor (observations x features)
            Constant scaling factors for mean model, such as library size factors.
        """
        # Define first layer of computation graph on identifiable variables
        # to yield dependent set of parameters of model for each location
        # and scale model.
        if constraints_loc is not None:
            a = self.apply_constraints(constraints=constraints_loc, var_all=a, dtype=dtype)

        if constraints_scale is not None:
            b = self.apply_constraints(constraints=constraints_scale, var_all=b, dtype=dtype)

        with tf.name_scope("mu"):
            log_mu = tf.matmul(design_loc, a, name="log_mu_obs")
            if size_factors is not None:
                log_mu = tf.add(log_mu, size_factors)
            log_mu = self.tf_clip_param(log_mu, "log_mu")
            mu = tf.exp(log_mu)

        with tf.name_scope("r"):
            log_r = tf.matmul(design_scale, b, name="log_r_obs")
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
