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
            "a_var": np.nextafter(-dmax, np.inf, dtype=dtype) / sf,
            "b_var": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "eta_loc": np.nextafter(-dmax, np.inf, dtype=dtype) / sf,
            "eta_scale": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "mean": np.nextafter(-dmax, np.inf, dtype=dtype) / sf,
            "sd": np.nextafter(0, np.inf, dtype=dtype),
            "probs": dtype(0),
            "log_probs": np.log(np.nextafter(0, np.inf, dtype=dtype)),
        }
        bounds_max = {
            "a_var": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "b_var": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "eta_loc": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "eta_scale": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "mean": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "sd": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
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
            a_var,
            b_var,
            dtype,
            size_factors=None
    ):
        a_var = self.tf_clip_param(a_var, "a_var")
        b_var = self.tf_clip_param(b_var, "b_var")

        if constraints_loc is not None:
            eta_loc = tf.matmul(design_loc, tf.matmul(constraints_loc, a_var))
        else:
            eta_loc = tf.matmul(design_loc, a_var)

        if size_factors is not None:
            eta_loc = tf.multiply(eta_loc, size_factors)

        eta_loc = self.tf_clip_param(eta_loc, "eta_loc")

        if constraints_scale is not None:
            eta_scale = tf.matmul(design_scale, tf.matmul(constraints_scale, b_var))
        else:
            eta_scale = tf.matmul(design_scale, b_var)

        eta_scale = self.tf_clip_param(eta_scale, "eta_scale")
        
        # Inverse linker functions:
        model_loc = eta_loc
        model_scale = tf.math.exp(eta_scale)

        # Log-likelihood:
        const = tf.constant(-0.5 * np.log(2 * np.pi), shape=(), dtype=dtype)
        if isinstance(X, tf.SparseTensor):
            log_probs = const - \
                        eta_scale - \
                        0.5 * tf.math.square(tf.divide(
                            tf.sparse.add(X, - model_loc),
                            model_scale
                        ))
            log_probs.set_shape([None, a_var.shape[1]])  # Need this so as shape is completely lost.
        else:
            log_probs = const - \
                        eta_scale - \
                        0.5 * tf.math.square(tf.divide(
                            X - model_loc,
                            model_scale
                        ))
        log_probs = self.tf_clip_param(log_probs, "log_probs")

        # Variance:
        sigma2 = tf.square(model_scale)

        self.X = X
        self.design_loc = design_loc
        self.design_scale = design_scale
        self.constraints_loc = constraints_loc
        self.constraints_scale = constraints_scale
        self.a_var = a_var
        self.b_var = b_var
        self.size_factors = size_factors
        self.dtype = dtype

        self.eta_loc = eta_loc
        self.eta_scale = eta_scale
        self.model_loc = model_loc
        self.model_scale = model_scale
        self.mean = model_loc
        self.sd = model_scale

        self.log_probs = log_probs

        self.sigma2 = sigma2
