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
            "a_var": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "b_var": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "eta_loc": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "eta_scale": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
            "p": np.nextafter(0, np.inf, dtype=dtype),
            "q": np.nextafter(0, np.inf, dtype=dtype),
            "probs": dtype(0),
            "log_probs": np.log(np.nextafter(0, np.inf, dtype=dtype)),
        }
        bounds_max = {
            "a_var": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "b_var": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "eta_loc": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "eta_scale": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
            "p": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "q": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "probs": np.nextafter(dmax, -np.inf, dtype=dtype) / sf,
            "log_probs": np.nextafter(np.log(dmax), -np.inf, dtype=dtype) / sf,
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

        eta_loc = self.tf_clip_param(eta_loc, "eta_loc")

        if constraints_scale is not None:
            eta_scale = tf.matmul(design_scale, tf.matmul(constraints_scale, b_var))
        else:
            eta_scale = tf.matmul(design_scale, b_var)

        eta_scale = self.tf_clip_param(eta_scale, "eta_scale")
        
        # Inverse linker functions:
        model_loc = tf.exp(eta_loc)
        model_scale = tf.exp(eta_scale)

        # Log-likelihood:
        if isinstance(X, tf.SparseTensor) or isinstance(X, tf.SparseTensorValue):
            Xdense = tf.sparse.to_dense(X)
            one_minus_X = - tf.sparse.add(X, -tf.ones(shape=X.dense_shape, dtype=self.dtype))
        else:
            Xdense = X
            one_minus_X = tf.ones_like(X)-X
        g = tf.lgamma(model_loc+model_scale) - tf.lgamma(model_loc) - tf.lgamma(model_scale)
        l = (model_loc - tf.ones_like(model_loc)) * tf.log(Xdense) + (model_scale - tf.ones_like(model_scale)) * tf.log(one_minus_X)
        log_probs = g + l

        a = tf.print("log_probs: \n", log_probs)

        log_probs = self.tf_clip_param(log_probs, "log_probs")

        b = tf.print("log_probs geclipt: \n", log_probs)
        c = tf.print("model_loc: \n", model_loc)
        d = tf.print("model_scale: \n", model_scale)
        # e = tf.print("first part: \n", h)
        # f = tf.print("second part: \n", i)

        # Variance:
        sigma2 = (model_loc * model_scale) \
                 / ((model_loc + model_scale)**2 * (model_loc + model_scale + tf.ones_like(model_scale)))

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
        self.p = model_loc
        self.q = model_scale

        with tf.control_dependencies([a,b,c,d]):
            self.log_probs = log_probs

        self.sigma2 = sigma2