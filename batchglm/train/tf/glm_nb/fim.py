import tensorflow as tf

import logging

from .external import FIMGLMALL

logger = logging.getLogger(__name__)


class FIM(FIMGLMALL):

    def _weight_fim_aa(
            self,
            loc,
            scale
    ):
        const = tf.divide(scale, scale + loc)
        W = tf.multiply(loc, const)

        return W

    def _weight_fim_bb(
            self,
            loc,
            scale
    ):
        return tf.zeros_like(scale)
        #scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        #scalar_two = tf.constant(2, shape=(), dtype=self.dtype)
        #scale_plus_loc = scale + loc
        #digamma_r = tf.math.digamma(x=scale)
        #digamma_r_plus_mu = tf.math.digamma(x=scale_plus_loc)
        #const1 = tf.multiply(scalar_two, tf.add(
        #    digamma_r,
        #    digamma_r_plus_mu
        #))
        #const2 = tf.multiply(scale, tf.add(
        #    tf.math.polygamma(a=scalar_one, x=scale),
        #    tf.math.polygamma(a=scalar_one, x=scale_plus_loc)
        #))
        #const3 = tf.divide(scale, scale_plus_loc)
        #W = tf.multiply(scale, tf.add_n([const1, const2, const3]))
        #return W
