import tensorflow as tf

import logging

from .external import FIMGLMALL

logger = logging.getLogger(__name__)


class FIM(FIMGLMALL):

    def _weight_fim_aa(
            self,
            mu,
            r
    ):
        const = tf.divide(r, r+mu)
        W = tf.multiply(mu, const)

        return W

    def _weight_fim_bb(
            self,
            X,
            mu,
            r
    ):
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        scalar_two = tf.constant(2, shape=(), dtype=self.dtype)

        r_plus_mu = r+mu
        digamma_r = tf.math.digamma(x=r)
        digamma_r_plus_mu = tf.math.digamma(x=r_plus_mu)

        const1 = tf.multiply(scalar_two, tf.add(
            digamma_r,
            digamma_r_plus_mu
        ))
        const2 = tf.multiply(r, tf.add(
            tf.math.polygamma(a=scalar_one, x=r),
            tf.math.polygamma(a=scalar_one, x=r_plus_mu)
        ))
        const3 = tf.divide(r, r_plus_mu)
        W = tf.multiply(r, tf.add_n([const1, const2, const3]))

        return W
