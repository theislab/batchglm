import tensorflow as tf
from .processModel import ProcessModel
from .external import LinearLocGLM, LinearScaleGLM, LinkerLocGLM
from .external import LinkerScaleGLM, LikelihoodGLM, UnpackParamsGLM


class UnpackParams(UnpackParamsGLM, ProcessModel):
    """
    Full class.
    """


class LinearLoc(LinearLocGLM, ProcessModel):

    def with_size_factors(self, eta_loc, size_factors):
        return tf.add(eta_loc, tf.math.log(size_factors))


class LinearScale(LinearScaleGLM, ProcessModel):
    """
    Full class.
    """


class LinkerLoc(LinkerLocGLM):

    def _inv_linker(self, loc: tf.Tensor):
        return tf.exp(loc)


class LinkerScale(LinkerScaleGLM):

    def _inv_linker(self, scale: tf.Tensor):
        return tf.exp(scale)


class Likelihood(LikelihoodGLM, ProcessModel):

    def _ll(self, eta_loc, eta_scale, loc, scale, x, n_features):

        # Log-likelihood:
        log_r_plus_mu = tf.math.log(scale + loc)
        if isinstance(x, tf.SparseTensor):
            log_probs_sparse = x.__mul__(eta_loc - log_r_plus_mu)
            log_probs_dense = tf.math.lgamma(tf.sparse.add(x, scale)) - \
                              tf.math.lgamma(tf.sparse.add(x, tf.ones(shape=x.dense_shape, dtype=self.ll_dtype))) - \
                              tf.math.lgamma(scale) + \
                              tf.multiply(scale, eta_scale - log_r_plus_mu)
            log_probs = tf.sparse.add(log_probs_sparse, log_probs_dense)
            # log_probs.set_shape([None, n_features])  # need as shape completely lost.
        else:
            log_probs = tf.math.lgamma(scale + x) - \
                        tf.math.lgamma(x + tf.ones_like(x)) - \
                        tf.math.lgamma(scale) + \
                        tf.multiply(x, eta_loc - log_r_plus_mu) + \
                        tf.multiply(scale, eta_scale - log_r_plus_mu)

        log_probs = self.tf_clip_param(log_probs, "log_probs")
        return log_probs
