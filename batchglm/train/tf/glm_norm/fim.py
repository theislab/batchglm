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
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        W = tf.square(tf.divide(scalar_one, scale))

        return W

    def _weight_fim_bb(
            self,
            loc,
            scale
    ):
        scalar_two = tf.constant(2, shape=(), dtype=self.dtype)
        W = scalar_two * tf.ones_like(loc)

        return W
