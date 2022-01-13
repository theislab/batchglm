import tensorflow as tf
import numpy as np
from .external import LinearLocGLM, LinearScaleGLM, LinkerLocGLM, LinkerScaleGLM, LikelihoodGLM, UnpackParamsGLM
from .processModel import ProcessModel


class UnpackParams(UnpackParamsGLM, ProcessModel):
    """
    Full class.
    """


class LinearLoc(LinearLocGLM, ProcessModel):

    def with_size_factors(self, eta_loc, size_factors):
        return tf.multiply(eta_loc, size_factors)


class LinearScale(LinearScaleGLM, ProcessModel):
    """
    Full Class
    """


class LinkerLoc(LinkerLocGLM):

    def _inv_linker(self, loc: tf.Tensor):
        return loc


class LinkerScale(LinkerScaleGLM):

    def _inv_linker(self, scale: tf.Tensor):
        return tf.math.exp(scale)


class Likelihood(LikelihoodGLM, ProcessModel):

    def _ll(self, eta_loc, eta_scale, loc, scale, x, n_features):

        const = tf.constant(-0.5 * np.log(2 * np.pi), shape=(), dtype=self.ll_dtype)
        if isinstance(x, tf.SparseTensor):
            log_probs = const - eta_scale - 0.5 * tf.math.square(tf.divide(tf.sparse.add(x, - loc), scale))
            # log_probs.set_shape([None, a_var.shape[1]])  # Need this so as shape is completely lost.
        else:
            log_probs = const - eta_scale - 0.5 * tf.math.square(tf.divide(x - loc, scale))
        log_probs = self.tf_clip_param(log_probs, "log_probs")

        return log_probs
