import logging

import tensorflow as tf

import numpy as np

from .external import ProcessModelGLM, ModelVarsGLM, BasicModelGraphGLM
from ..base_glm.model import ModelVarsEvalGLM
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
            "a_var": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "b_var": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "eta_loc": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "eta_scale": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "mu": np.nextafter(0, np.inf, dtype=dtype),
            "r": np.nextafter(0, np.inf, dtype=dtype),
            "probs": dtype(0),
            "log_probs": np.log(np.nextafter(0, np.inf, dtype=dtype)),
        }
        bounds_max = {
            "a_var": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "b_var": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "eta_loc": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "eta_scale": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
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


class ModelVarsEval(ProcessModel, ModelVarsEvalGLM):
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
            a_var,
            b_var,
            dtype,
            size_factors=None
    ):
        BasicModelGraphGLM.__init__(
            self=self,
            X=X,
            design_loc=design_loc,
            design_scale=design_scale,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            a_var=a_var,
            b_var=b_var,
            dtype=dtype,
            size_factors=size_factors
        )
        
        # Inverse linker functions:
        model_loc = tf.exp(self.eta_loc)
        model_scale = tf.exp(self.eta_scale)

        # Log-likelihood:
        log_r_plus_mu = tf.log(model_scale + model_loc)
        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            log_probs_sparse = X * (self.eta_loc - log_r_plus_mu)
            log_probs_dense = tf.math.lgamma(tf.sparse.add(X, model_scale)) - \
                              tf.math.lgamma(tf.sparse.add(X, tf.ones(shape=X.dense_shape, dtype=dtype))) + \
                              tf.math.lgamma(model_scale) + \
                              tf.multiply(model_scale, self.eta_scale - log_r_plus_mu)
            log_probs = tf.sparse.add(log_probs_sparse, log_probs_dense)
        else:
            log_probs = tf.math.lgamma(model_scale + X) - \
                        tf.math.lgamma(X + tf.ones_like(X)) - tf.math.lgamma(model_scale) + \
                        tf.multiply(X, self.eta_loc - log_r_plus_mu) + \
                        tf.multiply(model_scale, self.eta_scale - log_r_plus_mu)
        log_probs = self.tf_clip_param(log_probs, "log_probs")

        # Variance:
        sigma2 = model_loc + tf.multiply(tf.square(model_loc), model_scale)

        self.model_loc = model_loc
        self.model_scale = model_scale
        self.mu = model_loc
        self.r = model_scale

        self.log_probs = log_probs

        self.sigma2 = sigma2