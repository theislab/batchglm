import tensorflow as tf
from .external import LinearLocGLM, LinearScaleGLM, LinkerLocGLM, LinkerScaleGLM, LikelihoodGLM, UnpackParamsGLM
from .processModel import ProcessModel


class UnpackParams(UnpackParamsGLM, ProcessModel):
    """
    Full class.
    """


class LinearLoc(LinearLocGLM, ProcessModel):

    def with_size_factors(self, eta_loc, size_factors):
        raise NotImplementedError("There are no size_factors for GLMs with Beta noise.")


class LinearScale(LinearScaleGLM, ProcessModel):
    """
    Full Class
    """


class LinkerLoc(LinkerLocGLM):

    def _inv_linker(self, loc: tf.Tensor):
        return 1 / (1 + tf.exp(-loc))


class LinkerScale(LinkerScaleGLM):

    def _inv_linker(self, scale: tf.Tensor):
        return tf.exp(scale)


class Likelihood(LikelihoodGLM, ProcessModel):

    def _ll(self, eta_loc, eta_scale, loc, scale, x):

        if isinstance(x, tf.SparseTensor):
            one_minus_x = -tf.sparse.add(x, -tf.ones_like(loc))
        else:
            one_minus_x = 1 - x

        one_minus_loc = 1 - loc
        log_probs = tf.math.lgamma(scale) - tf.math.lgamma(loc * scale) \
                    - tf.math.lgamma(one_minus_loc * scale) \
                    + (scale * loc - 1) * tf.math.log(x) \
                    + (one_minus_loc * scale - 1) * tf.math.log(one_minus_x)

        log_probs = self.tf_clip_param(log_probs, "log_probs")

        return log_probs
